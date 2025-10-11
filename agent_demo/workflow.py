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
    # 使用 PostgresSaver.from_conn_string 作为上下文管理器
    connection_string = "postgresql://postgres:difyai123456@localhost:5432/postgres?sslmode=disable"

    async with AsyncPostgresSaver.from_conn_string(connection_string) as saver:
        # 确保表存在
        await saver.setup()
        print("✅ PostgresSaver initialized.")

        # 构建图workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("generate", generate_response)
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", END)

        agent1 = workflow.compile(checkpointer=saver)
        agent_supporter.init_agent(agent1)

        print("✅ Agent1 initialized.")

        # 在这里 yield，让 FastAPI 应用运行
        yield

        # 当应用关闭时，会回到这里，退出 with 块会自动清理连接
        print("🔌 PostgresSaver cleaned up.")

