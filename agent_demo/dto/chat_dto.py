from typing import Optional, List, Any

from openai import BaseModel

from agent_demo.const.chat_const import run_event


# 请求和响应模型
class ChatRequest(BaseModel):
    user_input: str
    kb_ids: Optional[List[str]] = None
    document_ids: Optional[List[str]] = None
    thread_id: Optional[str] = None


class Message(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    messages: List[Message]


class RunStreamResponse(BaseModel):
    event: str = run_event
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    answer: str = ""
    task_id: Optional[str] = None
    created_at: Optional[int] = None
    metadata: Optional[Any] = None
    usage_metadata: Optional[Any] = None
    code: Optional[int] = 0
    message: Optional[str] = None


class ContextSchema(BaseModel):
    user_input: str
    kb_ids: Optional[List[str]] = None
    document_ids: Optional[List[str]] = None
