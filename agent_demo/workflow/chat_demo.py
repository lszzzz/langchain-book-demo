import operator
from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agent_demo.workflow.workflow_base import WorkflowBase


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class ChatDemo(WorkflowBase):

    def __init__(self):
        super().__init__()

    async def generate_response(state: AgentState):
        """
        调用 LLM 生成响应
        """
        model = ChatDeepSeek(model="deepseek-chat")
        response = await model.ainvoke(state["messages"])

        # 将 AI 的回复添加到消息历史
        return {"messages": [response]}

    def compile(self, checkpointer):
        workflow = StateGraph(AgentState)
        workflow.add_node("generate", self.generate_response)
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", END)
        self.graph = workflow.compile(checkpointer)

    def get_graph(self):
        return self.graph


chat_demo_workflow = ChatDemo()
