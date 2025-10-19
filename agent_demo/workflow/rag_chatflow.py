import operator
from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langchain_community.chat_models import ChatTongyi


from agent_demo.workflow.workflow_base import WorkflowBase


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


async def generate_response(state: AgentState):
    """
    调用 LLM 生成响应
    """
    model = ChatTongyi(model="qwen-turbo")
    response = await model.ainvoke(state["messages"])

    # 将 AI 的回复添加到消息历史
    return {"messages": [response]}


class RagChatFlow(WorkflowBase):

    def __init__(self):
        super().__init__()

    def compile(self, checkpointer):
        workflow = StateGraph(AgentState)
        workflow.add_node("generate", generate_response)
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", END)
        self.graph = workflow.compile(checkpointer)


rag_chatflow = RagChatFlow()
