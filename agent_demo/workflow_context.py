# workflow.py
from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from agent_demo.workflow.chat_demo import chat_demo_workflow
from agent_demo.workflow.chat_tools_demo import chat_tools_demo_workflow
from agent_demo.workflow.rag_chatflow import rag_chatflow


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
        print("✅ chat_demo_workflow initialized.")

        chat_tools_demo_workflow.compile(checkpointer=saver)
        print("✅ chat_tools_demo_workflow initialized.")

        rag_chatflow.compile(checkpointer=saver)
        print("✅ rag_chatflow initialized.")

        # 在这里 yield，让 FastAPI 应用运行
        yield

        # 当应用关闭时，会回到这里，退出 with 块会自动清理连接
        print("🔌 PostgresSaver cleaned up.")
