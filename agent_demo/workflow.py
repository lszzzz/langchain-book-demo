# workflow.py
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage


# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_input: str


# 定义节点函数
def generate_response(state: AgentState):
    """
    调用 LLM 生成响应
    """
    model = ChatDeepSeek(model="deepseek-chat")

    # 构建消息列表，可以包含系统提示
    # messages = [
    #                SystemMessage(content="你是一个友好的AI助手，请用中文回答。"),
    #            ] + state["messages"]

    response = model.invoke(state["messages"])

    # 将 AI 的回复添加到消息历史
    return {"messages": [response]}


def user_input_node(state: AgentState):
    """
    处理用户输入，可以在这里做预处理
    """
    user_msg = HumanMessage(content=state["user_input"])
    return {"messages": [user_msg]}


# 构建图
def create_agent():
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("user_input", user_input_node)
    workflow.add_node("generate", generate_response)

    # 设置入口节点
    workflow.set_entry_point("user_input")

    # 定义边：从 user_input 到 generate，然后到 END
    workflow.add_edge("user_input", "generate")
    workflow.add_edge("generate", END)

    # 编译图
    app = workflow.compile()
    return app


# 创建可调用的工作流实例
agent = create_agent()
