import os
from openai import OpenAI
import math
import json

# Установите ваш API ключ OpenAI
client = OpenAI(
    api_key="kek",
    base_url="http://81.94.156.140:8999/v1",
)

# Определение функции калькулятора
def calculator(expression):
    """
    Вычисляет математическое выражение, переданное в виде строки.
    Поддерживает основные математические операции и функции из модуля math.
    """
    try:
        # Для безопасного вычисления используем ограниченный набор операций
        allowed_names = {
            k: v for k, v in math.__dict__.items() 
            if not k.startswith('__')
        }
        
        # Добавляем основные операции
        allowed_names.update({
            'abs': abs,
            'float': float,
            'int': int,
            'max': max,
            'min': min,
            'round': round,
            'sum': sum
        })
        
        # Вычисляем выражение
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return result
    except Exception as e:
        return f"Ошибка при вычислении: {str(e)}"

# Функция для создания запроса к OpenAI с поддержкой функции калькулятора
def ask_openai(query):
    # Определяем доступные функции
    functions = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Вычисляет результат математического выражения",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Математическое выражение для вычисления"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    try:
        # Отправляем запрос к OpenAI
        response = client.chat.completions.create(
            model="LLM", 
            messages=[{"role": "user", "content": query}],
            tools=functions,
            tool_choice="auto",
            temperature=0.1,
        )
        
        message = response.choices[0].message
        
        # Проверяем, есть ли вызов функции в ответе
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print("ДААААААААААААААААААААААА")
            tool_call = message.tool_calls[0]
            if tool_call.function.name == "calculator":
                # Извлекаем выражение из аргументов функции
                function_args = json.loads(tool_call.function.arguments)
                expression = function_args.get("expression")
                
                # Вычисляем результат
                calculation_result = calculator(expression)
                
                # Отправляем результат обратно к OpenAI
                final_response = client.chat.completions.create(
                    model="LLM",
                    messages=[
                        {"role": "user", "content": query},
                        message,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "calculator",
                            "content": str(calculation_result)
                        },
                    ],
                    temperature=0.1
                )
                return final_response.choices[0].message.content
            
        # Если нет вызова функции, возвращаем обычный ответ
        return message.content
    
    except Exception as e:
        return f"Произошла ошибка при запросе к OpenAI: {str(e)}"

# Пример использования
if __name__ == "__main__":
    while True:
        user_query = input("Введите ваш вопрос (или 'выход' для завершения): ")
        if user_query.lower() == 'выход':
            break
        
        response = ask_openai(user_query)
        print(f"Ответ: {response}")