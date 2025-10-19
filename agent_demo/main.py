# main.py
import json
import logging
import traceback
import uuid
from fastapi import FastAPI
from langchain_core.messages import HumanMessage

from sse_starlette import EventSourceResponse
from starlette import status

from agent_demo.dto.chat_dto import ChatResponse, ChatRequest, Message, RunStreamResponse
from agent_demo.workflow.chat_demo import chat_demo_workflow
from agent_demo.workflow.chat_tools_demo import chat_tools_demo_workflow
from agent_demo.workflow.rag_chatflow import rag_chatflow
from workflow_context import lifespan_context

app = FastAPI(title="LangGraph + FastAPI 示例", lifespan=lifespan_context)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    接收用户输入，调用 LangGraph 工作流，返回 AI 响应
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 调用 LangGraph 工作流
    result = await chat_demo_workflow.get_graph().ainvoke(
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
            async for message_chunk, metadata in chat_demo_workflow.get_graph().astream(
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
    result = await chat_tools_demo_workflow.get_graph().ainvoke(
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


@app.post("/rag_chatflow", response_model=ChatResponse)
async def rag_chatflow_endpoint(request: ChatRequest):
    """
    接收用户输入，调用 LangGraph 工作流，返回 AI 响应
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    context = {
        "user_input": request.user_input,
        "kb_ids": request.kb_ids,
        "document_ids": request.document_ids,
    }

    # 调用 LangGraph 工作流
    result = await rag_chatflow.get_graph().ainvoke(
        input={"messages": [HumanMessage(content=request.user_input)]},
        config=config,
        context=context
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
    uvicorn.run(app, host="localhost", port=8200)
