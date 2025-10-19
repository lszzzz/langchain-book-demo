import operator
from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langchain_community.chat_models import ChatTongyi


from agent_demo.workflow.workflow_base import WorkflowBase


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# async def intent_recognition(state: AgentState):
#     if not state["kb_ids"] and not state["document_ids"]:
#         return "direct_llm"
#
#     # 调用llm分析
#     prompt = '''
#     你是Microcraft，是由Microware科技公司开发和提供的人工智能助手。
#     ## 目标
#     在确保内容安全合规的情况下通过遵循指令和提供有帮助回复来帮助用户实现他们的目标。
#
#     辨别用户的问题是否可能需要调用内部知识库工具来检索回答：
#     如果用户的问题是很常见的问题，例如“你好/你是谁/今天天气如何/”等在大模型知识范围内的问题，请输出 -> 否。否则可能需要调用知识库检索，请输出-> 是
#     不要给出多余解释，直接回答 是 或 否
#     '''
#
#     model = ChatTongyi(model="qwen-turbo")
#     response = await model.ainvoke(state["messages"])
#
#     # 将 AI 的回复添加到消息历史
#     return {"messages": [response]}


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
