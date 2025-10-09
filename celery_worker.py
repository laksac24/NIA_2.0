from celery import Celery
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from guidance import guidance
import os
from dotenv import load_dotenv
import asyncio
import ssl

load_dotenv()

# Celery configuration
# REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
# REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_BROKER_URL = os.getenv("REDIS_BROKER_URL")
REDIS_RESULT_BACKEND = os.getenv("REDIS_RESULT_BACKEND")

# Initialize Celery app
celery_app = Celery(
    "guidance_worker",
    broker=REDIS_BROKER_URL,
    backend=REDIS_RESULT_BACKEND
)

celery_app.conf.broker_use_ssl = {'ssl_cert_reqs': ssl.CERT_NONE}
celery_app.conf.redis_backend_use_ssl = {'ssl_cert_reqs': ssl.CERT_NONE}

# Celery settings
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    task_soft_time_limit=240,
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str


# Initialize LLM (same as main.py)
llm = ChatGroq(model="openai/gpt-oss-20b")


async def decision_llm(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Tell me whether the given user query is:
             - a normal query, like asking for information, advice, etc, or something not related to guidance or test (in that case, just say "normal")
             - guidance related (like regarding future plans or scope in different fields or he wants to explore some field, etc)
             return the answer in only one word, nothing else, either guidance or normal
             """),
            ("user", "command: {command}")
        ]
    )

    user_input = state["user_input"]
    chain = prompt | llm
    response = await chain.ainvoke({"command": user_input})
    return {"messages": state["messages"] + [response]}


def decision(state: State):
    dec = state["messages"][-1].content.lower().strip()
    if "guidance" in dec:
        return "guidance_node"
    elif "normal" in dec:
        return "normal_node"


async def guidance_llm(state: State):
    user_input = state["user_input"]
    output = await guidance(user_input)
    return {"messages": state["messages"] + [output]}


async def normal_llm(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful assistant. Answer the user query as best as you can, just answer point to point, nothing else, and dont add any ** or something to show bold letters and all. Dont extend the answer unnecessarily, keep it short and precise.
             """),
            ("user", "command: {command}")
        ]
    )
    user_input = state["user_input"]
    chain = prompt | llm
    response = await chain.ainvoke({"command": user_input})
    return {"messages": state["messages"] + [response]}


# Build the same graph as in main.py
graph_builder = StateGraph(State)

#nodes
graph_builder.add_node("decision_node", decision_llm)
graph_builder.add_node("guidance_node", guidance_llm)
graph_builder.add_node("normal_node", normal_llm)

#edges
graph_builder.set_entry_point("decision_node")
graph_builder.add_conditional_edges(
    "decision_node",
    decision,
    {
        "guidance_node": "guidance_node",
        "normal_node": "normal_node"
    }
)
graph_builder.add_edge("guidance_node", END)
graph_builder.add_edge("normal_node", END)

graph = graph_builder.compile()

@celery_app.task(bind=True, name="process_query", max_retries=3)
def process_query(self, query: str):
    try:
        loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)
        state = {
            "messages": [{"role": "user", "content": query}],
            "user_input": query
        }

        response = loop.run_until_complete(graph.ainvoke(state))
        final_response = response["messages"][-1].content
        loop.close()

        return {
            "query": query,
            "response": final_response,
            "status": "success"
        }
    
    except Exception as e:
        try:
            raise self.retry(exc=e, countdown=5)
        except self.MaxRetriesExceededError:
            return {
                "query": query,
                "error": str(e),
                "status": "failed"
            }