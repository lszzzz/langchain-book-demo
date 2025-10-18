# workflow.py
from contextlib import asynccontextmanager

from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import START
from langgraph.graph import StateGraph, END, MessagesState
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode

from agent_demo.agent_support import agent_supporter
from agent_demo.workflow.chat_demo import chat_demo_workflow


# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


''' agent2 '''
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."


tool_node = ToolNode([get_weather])

agent2_model = ChatDeepSeek(model="deepseek-chat")
agent2_model_with_tools = agent2_model.bind_tools([get_weather])


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = agent2_model_with_tools.invoke(messages)
    print("response:{}".format(response))
    return {"messages": [response]}


@asynccontextmanager
async def lifespan_context(app):
    # 使用 PostgresSaver.from_conn_string 作为上下文管理器
    connection_string = "postgresql://postgres:difyai123456@localhost:5432/postgres?sslmode=disable"

    async with AsyncPostgresSaver.from_conn_string(connection_string) as saver:
        # 确保表存在
        await saver.setup()
        print("✅ PostgresSaver initialized.")

        # 构建chat_demo_agent
        chat_demo_workflow.compile(checkpointer=saver)

        print("✅ Agent1 initialized.")

        # 构建agent2
        builder = StateGraph(MessagesState)

        # Define the two nodes we will cycle between
        builder.add_node("call_model", call_model)
        builder.add_node("tools", tool_node)

        builder.add_edge(START, "call_model")
        builder.add_conditional_edges("call_model", should_continue, ["tools", END])
        builder.add_edge("tools", "call_model")

        agent2 = builder.compile(checkpointer=saver)
        agent_supporter.init_agent2(agent2)

        print("✅ Agent2 initialized.")

        # 在这里 yield，让 FastAPI 应用运行
        yield

        # 当应用关闭时，会回到这里，退出 with 块会自动清理连接
        print("🔌 PostgresSaver cleaned up.")
