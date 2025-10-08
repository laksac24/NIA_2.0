from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from user_test import user_test
from guidance import guidance
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
import hashlib
import json
from celery.result import AsyncResult
from celery_worker import celery_app, process_query
# from celery import Celery
load_dotenv()


class State(TypedDict):
    messages : Annotated[list,add_messages]
    user_input : str

# celery_app = Celery(
#     "tasks",
#     broker = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
#     backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
# )

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600)) 

redis_client = None

async def get_redis_client():
    global redis_client
    if redis_client is None:
        try:
            redis_client = await redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True
            )
            
            await redis_client.ping()
            print("Redis connected successfully")
        except Exception as e:
            print(f"Redis connection failed: {e}")
            print("App will run without caching")
            redis_client = None

    return redis_client

def generate_cache_key(query: str) -> str:
    normalised_query = query.lower().strip()
    return f"query:{hashlib.sha256(normalised_query.encode()).hexdigest()}"

async def get_cached_response(query: str):
    r = await get_redis_client()
    cache_key = generate_cache_key(query)
    cached = await r.get(cache_key)
    if cached:
        return json.loads(cached)
    return None

async def set_cached_response(query:str, response: str):
    r = await get_redis_client()
    cache_key = generate_cache_key(query)
    await r.setex(
        cache_key,
        CACHE_TTL,
        json.dumps(response)
    ) 
    

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

    chain = prompt|llm
    response = await chain.ainvoke({"command" : user_input})
    return {"messages" : state["messages"] + [response]}

def decision(state:State):
    dec = state["messages"][-1].content.lower().strip()
    if "guidance" in dec:
        return "guidance_node"
    elif "normal" in dec:
        return "normal_node"
    
async def guidance_llm(state:State):
    user_input = state["user_input"]
    output = await guidance(user_input)

    return {"messages" : state["messages"] + [output]}

async def normal_llm(state:State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful assistant. Answer the user query as best as you can, just answer point to point, nothing else, and dont add any ** or something to show bold letters and all. Dont extend the answer unnecessarily, keep it short and precise.
             """),
             ("user", "command: {command}")
        ]
    )
    user_input = state["user_input"]
    chain = prompt|llm
    response = await chain.ainvoke({"command" : user_input})
    return {"messages" : state["messages"] + [response]}

graph_builder = StateGraph(State)

#nodes
graph_builder.add_node("decision_node",decision_llm)
graph_builder.add_node("guidance_node",guidance_llm)
graph_builder.add_node("normal_node",normal_llm)

#edges
graph_builder.set_entry_point("decision_node")
graph_builder.add_conditional_edges(
    "decision_node",
    decision,
    {
        "guidance_node":"guidance_node",
        "normal_node":"normal_node"
    }
)
graph_builder.add_edge("guidance_node",END)
graph_builder.add_edge("normal_node",END)

graph = graph_builder.compile()

app = FastAPI(title="Guidance & Test API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_evemt():
    await get_redis_client()

@app.on_event("shutdown")
async def shutdown_event():
    global redis_client
    if redis_client:
        await redis_client.close()



@app.post("/ask")
async def ask_bot(request: UserRequest):

    cache_response = await get_cached_response(request.query)
    if cache_response:
        return {"response": cache_response, "cached": True}

    state = {"messages": [{"role": "user", "content": request.query}], "user_input": request.query}
    response = await graph.ainvoke(state)
    final_response = response["messages"][-1].content

    await set_cached_response(request.query, final_response)

    return {"response": final_response, "cached": False}

@app.post("/ask/async")
async def ask_bot_async(request: UserRequest):

    cache_response = await get_cached_response(request.query)
    if cache_response:
        return {"response": cache_response, "cached": True, "status": "completed"}
    
    task = process_query.delay(request.query)

    return {
        "task_id": task.id,
        "status": "processing",
        "message": "Query submitted for processing. Use /task/{task_id} to check status"
    }

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app = celery_app)
    if task_result.ready():
        if task_result.successful():
            result = task_result.get()
            # Cache the result
            if "query" in result and "response" in result:
                await set_cached_response(result["query"], result["response"])
            return {
                "task_id": task_id,
                "status": "completed",
                "response": result.get("response"),
                "cached": False
            }
        else:
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(task_result.info)
            }
    else:
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Task is still being processed"
        }

@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cache entries"""
    try:
        r = await get_redis_client()
        keys = await r.keys("query:*")
        if keys:
            await r.delete(*keys)
        return {"message": f"Cleared {len(keys)} cache entries"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    try:
        r = await get_redis_client()
        keys = await r.keys("query:*")
        return {"total_cached_queries": len(keys)}
    except Exception as e:
        return {"error": str(e)}