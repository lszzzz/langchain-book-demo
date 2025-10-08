# main.py
import uuid

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from workflow import agent

app = FastAPI(title="LangGraph + FastAPI 示例")


# 请求和响应模型
class ChatRequest(BaseModel):
    user_input: str
    thread_id: Optional[str] = None


class Message(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    messages: List[Message]


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    接收用户输入，调用 LangGraph 工作流，返回 AI 响应
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 调用 LangGraph 工作流
    result = await agent.ainvoke(
        input={"messages": [HumanMessage(content=request.user_input)]},
        config=config,
    )

    # 提取 AI 的响应
    ai_message = result["messages"][-1]  # 最后一条是 AI 的回复
    response_text = ai_message.content

    # 转换消息历史为 API 响应格式
    message_history = []
    for msg in result["messages"]:
        role = msg.type
        content = msg.content
        message_history.append(Message(role=role, content=content))

    return ChatResponse(
        thread_id=thread_id,
        response=response_text,
        messages=message_history
    )


@app.get("/")
async def root():
    return {"message": "LangGraph + FastAPI 服务已启动！访问 /docs 查看 API 文档。"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
