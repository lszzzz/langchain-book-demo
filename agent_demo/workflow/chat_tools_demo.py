from langchain_deepseek import ChatDeepSeek
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agent_demo.workflow.workflow_base import WorkflowBase


def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."


tool_node = ToolNode([get_weather])

llm_model = ChatDeepSeek(model="deepseek-chat")
model_with_tools = llm_model.bind_tools([get_weather])


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    print("response:{}".format(response))
    return {"messages": [response]}


class ChatToolsDemo(WorkflowBase):

    def __init__(self):
        super().__init__()

    def compile(self, checkpointer):
        # 构建agent2
        workflow = StateGraph(MessagesState)

        # Define the two nodes we will cycle between
        workflow.add_node("call_model", call_model)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "call_model")
        workflow.add_conditional_edges("call_model", should_continue, ["tools", END])
        workflow.add_edge("tools", "call_model")

        self.graph = workflow.compile(checkpointer=checkpointer)


chat_tools_demo_workflow = ChatToolsDemo()
