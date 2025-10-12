# main.py
import json
import logging
import traceback
import uuid
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal

from sse_starlette import EventSourceResponse
from starlette import status

from agent_demo.agent_support import agent_supporter
from workflow import lifespan_context

app = FastAPI(title="LangGraph + FastAPI 示例", lifespan=lifespan_context)


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
    result = await agent_supporter.get_agent().ainvoke(
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


run_event = Literal["agent_message", "message_end", "error", "new_conversation"]


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


@app.post("/chat_stream", status_code=status.HTTP_200_OK, response_model_exclude_none=True)
async def chat_stream_endpoint(request: ChatRequest):
    """
    接收用户输入，调用 LangGraph 工作流，返回 AI 响应
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            # 调用 LangGraph 工作流
            async for message_chunk, metadata in agent_supporter.get_agent().astream(
                    input={"messages": [HumanMessage(content=request.user_input)]},
                    config=config,
                    stream_mode="messages",
            ):
                if message_chunk.content:
                    resp = RunStreamResponse(
                        event="agent_message",
                        answer=message_chunk.content
                    )

                    result = resp.model_dump_json()
                    yield result
                    logging.info(f"web chat stream send: {result}")

        except Exception as e:
            logging.error("run/stream exception:{}".format(traceback.format_exc()))
            data = json.dumps({
                "event": "error",
                "code": 99999,
                "message": "system error"
            })
            yield data

    return EventSourceResponse(event_generator())



@app.post("/chat_tools", response_model=ChatResponse)
async def chat_tools_endpoint(request: ChatRequest):
    """
    接收用户输入，调用 LangGraph 工作流，返回 AI 响应
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 调用 LangGraph 工作流
    result = await agent_supporter.get_agent2().ainvoke(
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
