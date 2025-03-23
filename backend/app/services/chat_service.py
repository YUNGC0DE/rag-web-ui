import json
import base64
import logging
from typing import List, AsyncGenerator, Dict, Any
from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
from app.models.chat import Message
from app.models.knowledge import KnowledgeBase, Document
from langchain.globals import set_verbose, set_debug
from app.services.vector_store.chroma import ChromaVectorStore
from app.services.llm.llm_factory import LLMFactory
from app.services.embedding.embedding_factory import EmbeddingsFactory
from app.services.vector_store import VectorStoreFactory

#set_verbose(True)
#set_debug(True)

async def generate_response(
    query: str,
    messages: dict,
    knowledge_base_ids: List[int],
    chat_id: int,
    db: Session
) -> AsyncGenerator[str, None]:
    try:
        # Create user message
        user_message = Message(
            content=query,
            role="user",
            chat_id=chat_id
        )
        db.add(user_message)
        db.commit()
        
        # Create bot message placeholder
        bot_message = Message(
            content="",
            role="assistant",
            chat_id=chat_id
        )
        db.add(bot_message)
        db.commit()
        
        # Get knowledge bases and their documents
        knowledge_bases = (
            db.query(KnowledgeBase)
            .filter(KnowledgeBase.id.in_(knowledge_base_ids))
            .all()
        )
        
        # Initialize embeddings
        embeddings = EmbeddingsFactory.create()
        embeddings.set_task("retrieval.query")
        kb = knowledge_bases[0] # TODO: In case we will need other knowledge bases
        retriever = ChromaVectorStore(collection_name=f"kb_{kb.id}", embedding_function=embeddings).as_retriever()
        
        # Initialize the language model
        llm = LLMFactory.create()
        
        # Преобразуем историю сообщений
        chat_history = []
        # TODO move to config
        NUMBER_OF_MESSAGES_TO_INCLUDE_IN_CONTEXT = 4
        for message in messages["messages"][-NUMBER_OF_MESSAGES_TO_INCLUDE_IN_CONTEXT-1:-1]:
            if message["role"] == "user":
                chat_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                # if include __LLM_RESPONSE__, only use the last part
                if "__LLM_RESPONSE__" in message["content"]:
                    message["content"] = message["content"].split("__LLM_RESPONSE__")[-1]
                chat_history.append(AIMessage(content=message["content"]))
        
        # 1. ЭТАП ПРОМПТ ЭНРИЧИНГА - улучшенный промпт
        # TODO FIX THIS SHIT
        query_enrichment_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Дополни последний вопрос (сообщение) пользователя информацией из предыдущих сообщений для получения наиболее эффективного запроса для поиска информации в базе знаний.
            Если вопрос никак не связан с контекстом или его не надо улучшать, то верни исходный вопрос без изменений.
            Не пиши никакой другой текст, не придумывай информацию, не отвечай на вопрос и не выдавай свои мысли.
            Твой ответ должен состоять только из улучшенного вопроса (сообщения), либо из вопроса (сообщения) без изменений.
            """),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])
        
        # 2. ЭТАП ГЕНЕРАЦИИ ОТВЕТА - QA prompt
        qa_system_prompt = """
        Ты - высококвалифицированный ассистент, который дает точные, информативные и полезные ответы, опираясь только на предоставленный контекст.
        
        КОНТЕКСТ:
        {context}
        
        ИНСТРУКЦИИ:
        1. Ответь на вопрос пользователя, опираясь ТОЛЬКО на информацию из предоставленного контекста
        2. Цитируй источники в формате [citation:X], где X - номер источника (начиная с 1)
        3. Если в одном предложении используется информация из нескольких источников, укажи все: [citation:1][citation:2]
        4. Если контекст не содержит достаточно информации, напиши "информация отсутствует по" и укажи тему
        5. Твой ответ должен быть профессиональным, точным и непредвзятым
        6. Отвечай на языке вопроса (кроме кода и специфических терминов)
        7. Будь лаконичным, но информативным
        
        ВАЖНО: Не придумывай информацию. Цитируй только из предоставленного контекста.
        """
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])
        
        # Определяем функции для каждого этапа RAG цепочки
        
        def enrich_query(x):
            enriched_question = query_enrichment_prompt | llm.with_config({"run_name": "query_enrichment"}) | StrOutputParser()
            return {
                "chat_history": x["chat_history"],
                "question": x["question"],
                "enriched_question": enriched_question.invoke({"question": x["question"], "chat_history": x["chat_history"]})
            }
        
        def retrieve_docs(x):
            docs = retriever.get_relevant_documents(x["enriched_question"])
            return {
                "chat_history": x["chat_history"],
                "question": x["question"],
                "documents": docs,
            }
        
        async def generate_answer(x):
            answer_chain = qa_prompt | llm.with_config({"run_name": "answer_generation"}) | StrOutputParser()
            
            # Отдаем документы
            yield {
                "documents": x["documents"]
            }   

            # Запускаем стриминг ответа
            async for chunk in answer_chain.astream({
                "context": x["documents"],
                "question": x["question"],
                "chat_history": x["chat_history"]
            }):
                yield {
                    "answer": chunk
                }
        
        # Создаем цепочку из трех шагов с использованием pipe
        query_enrichment = RunnableLambda(enrich_query)
        document_retrieval = RunnableLambda(retrieve_docs)
        answer_generation = RunnableLambda(generate_answer)
        
        rag_chain = query_enrichment.pipe(document_retrieval).pipe(answer_generation)
        
        # Логирование для отладки
        logging.info("Starting RAG chain processing")
        
        # Генерация ответа через созданную цепь
        full_response = ""

        # Вызываем цепь с начальными данными
        async for chunk in rag_chain.astream({
            "question": query,
            "chat_history": chat_history
        }):
            # Проверка типа и содержимого chunk
            if not isinstance(chunk, dict):
                logging.warning(f"Unexpected chunk type: {type(chunk)}")
                continue
            # Отправка контекста на фронтенд (только один раз)
            if "documents" in chunk:
                documents = chunk["documents"]
                serializable_context = []
                
                for i, context in enumerate(documents):
                    serializable_doc = {
                        "id": context.metadata.get("chunk_id", ""),
                        "kb_id": context.metadata.get("kb_id", ""),
                        "document_id": context.metadata.get("document_id", ""),
                        "metadata": {
                            k: v for k, v in context.metadata.items() 
                            if k not in ["chunk_id", "kb_id", "document_id"]
                        }
                    }
                    serializable_context.append(serializable_doc)
                
                # Логирование промежуточных результатов
                logging.info(f"Retrieved {len(documents)} documents")
                
                escaped_context = json.dumps({
                    "context_refs": serializable_context
                })
                
                base64_context = base64.b64encode(escaped_context.encode()).decode()
                separator = "__LLM_RESPONSE__"
                
                yield f'0:"{base64_context}{separator}"\n'
                full_response += base64_context + separator
            
            # Стриминг ответа - теперь каждый чанк содержит непосредственно часть ответа
            if "answer" in chunk:
                answer_chunk = chunk["answer"]
                if answer_chunk:
                    # Логирование для отладки
                    logging.debug(f"Received new chunk: {answer_chunk}")
                    
                    full_response += answer_chunk
                    # Безопасное экранирование строки
                    safe_chunk = json.dumps(answer_chunk)[1:-1]  # Используем json.dumps для безопасного экранирования
                    yield f'0:"{safe_chunk}"\n'
        
        # Обновляем содержимое сообщения бота
        bot_message.content = full_response
        db.commit()
            
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        logging.exception(e, exc_info=True, stack_info=True)
        yield '3:{text}\n'.format(text=error_message)
        
        # Обновляем сообщение бота с ошибкой
        if 'bot_message' in locals():
            bot_message.content = error_message
            db.commit()
    finally:
        db.close()