# workflow.py
from contextlib import asynccontextmanager

from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import START
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from agent_demo.agent_support import agent_supporter


# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# 定义节点函数
async def generate_response(state: AgentState):
    """
    调用 LLM 生成响应
    """
    model = ChatDeepSeek(model="deepseek-chat")
    response = await model.ainvoke(state["messages"])

    # 将 AI 的回复添加到消息历史
    return {"messages": [response]}


@asynccontextmanager
async def lifespan_context(app):
    # 1. 使用 PostgresSaver.from_conn_string 作为上下文管理器
    connection_string = "postgresql://postgres:difyai123456@localhost:5432/postgres?sslmode=disable"

    async with AsyncPostgresSaver.from_conn_string(connection_string) as saver:
        # 2. 确保表存在
        await saver.setup()

        # 3. 构建图
        workflow = StateGraph(AgentState)
        workflow.add_node("generate", generate_response)
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", END)

        # 4. 编译图
        app = workflow.compile(checkpointer=saver)
        agent_supporter.init_agent(app)

        print("✅ Agent and PostgresSaver initialized.")

        # 在这里 yield，让 FastAPI 应用运行
        yield

        # 当应用关闭时，会回到这里，退出 with 块会自动清理连接
        print("🔌 PostgresSaver cleaned up.")

