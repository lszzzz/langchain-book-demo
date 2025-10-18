from abc import ABC, abstractmethod

from langgraph.graph import StateGraph


class WorkflowBase(ABC):
    def __init__(self):
        self.graph = None

    @abstractmethod
    def compile(self, checkpointer) -> StateGraph:
        raise NotImplementedError
