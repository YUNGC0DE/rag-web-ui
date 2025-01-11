from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class DocumentBase(BaseModel):
    title: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    content_type: Optional[str] = None

class DocumentCreate(DocumentBase):
    knowledge_base_id: int

class DocumentResponse(DocumentBase):
    id: int
    knowledge_base_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class KnowledgeBaseBase(BaseModel):
    name: str
    description: Optional[str] = None

class KnowledgeBaseCreate(KnowledgeBaseBase):
    pass

class KnowledgeBaseUpdate(KnowledgeBaseBase):
    pass

class KnowledgeBaseResponse(KnowledgeBaseBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    documents: List[DocumentResponse] = []

    class Config:
        from_attributes = True 