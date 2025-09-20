from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
# from input import get_input
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from user_test import user_test
from guidance import guidance
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()


class State(TypedDict):
    messages : Annotated[list,add_messages]
    user_input : str

llm = ChatGroq(model="Gemma2-9b-It")

tool = TavilySearch(max_results=2)
tools = [tool]

def decision_llm(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Tell me whether the given user query is:
             - a normal query, like asking for information, advice, etc, or something not related to guidance or test (in that case, just say "normal")
             - guidance related (like regarding future plans or scope in different fields or he wants to explore some field, etc)
             - test related (whether the user wants to check is knowledge in a perticular field through a test or something)
             return the answer in only one word, nothing else, either guidance or test
             """),
             ("user", "command: {command}")
        ]
    )

    user_input = state["user_input"]

    chain = prompt|llm
    response = chain.invoke({"command" : user_input})
    return {"messages" : state["messages"] + [response]}

def decision(state:State):
    dec = state["messages"][-1].content.lower().strip()
    if "guidance" in dec:
        return "guidance_node"
    elif "test" in dec:
        return "test_node"
    elif "normal" in dec:
        return "normal_node"
    
def guidance_llm(state:State):
    user_input = state["user_input"]
    output = guidance(user_input)

    return {"messages" : state["messages"] + [output]}

def test_llm(state:State):
    user_input = state["user_input"]
    output = user_test(user_input)

    return {"messages" : state["messages"] + [output]}

def normal_llm(state:State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful assistant. Answer the user query as best as you can, just answer point to point, nothing else, and dont add any ** or something to show bold letters and all. Dont extend the answer unnecessarily, keep it short and precise.
             """),
             ("user", "command: {command}")
        ]
    )
    user_input = state["user_input"]
    chain = prompt|llm
    response = chain.invoke({"command" : user_input})
    return {"messages" : state["messages"] + [response]}

graph_builder = StateGraph(State)

#nodes
graph_builder.add_node("decision_node",decision_llm)
graph_builder.add_node("guidance_node",guidance_llm)
graph_builder.add_node("test_node",test_llm)
graph_builder.add_node("normal_node",normal_llm)

#edges
graph_builder.set_entry_point("decision_node")
graph_builder.add_conditional_edges(
    "decision_node",
    decision,
    {
        "guidance_node":"guidance_node",
        "test_node":"test_node",
        "normal_node":"normal_node"
    }
)

graph = graph_builder.compile()

# while True:

#     # input_type, content = get_input()
#     content = input()

#     # if input_type == "text":
#     #     print(f"You typed: {content}")
#     # elif input_type == "voice":
#     #     print(f"You said: {content}")
#     # elif input_type == "quit":
#     #     print("User wants to exit")
#     # elif input_type == "error":
#     #     print(f"Error: {content}")
    
    
#     if not content:
#         continue
#     state = {
#         "messages": [{"role": "user", "content": content}],
#         "user_input" : content
#     }
#     response = graph.invoke(state)
#     print(response["messages"][-1].content)


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


@app.post("/ask")
def ask_bot(request: UserRequest):
    state = {"messages": [{"role": "user", "content": request.query}], "user_input": request.query}
    response = graph.invoke(state)
    return {"response": response["messages"][-1].content}