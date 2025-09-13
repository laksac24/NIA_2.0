from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate

class State(TypedDict) :
    messages:Annotated[list,add_messages]

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm=ChatGroq(model="llama-3.3-70b-versatile")

from langchain_tavily import TavilySearch
tool = TavilySearch(max_results=2, time_range="month")

tools=[tool]

llm_with_tools=llm.bind_tools(tools)

from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

def tool_calling_llm(state:State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
                You are an expert career mentor and advisor for students.

                Your job is to guide students based on their queries by providing:
                - A brief overview of the field or topic they mentioned. (only where you think its required)
                - Current trends and technologies in that domain.
                - Future career opportunities and job roles.
                - Related fields they can explore.
                - Practical next steps (like courses, skills to build, or resources).

                Guidelines:
                - Dont add any ** or something to show bold letters and all
                - Keep the response short, NOT AI generated but well-structured, properly formatted and easy to read.
                - Use points where possible.
                - Use friendly but professional tone.
                - If the answer requires external information (like latest trends or news), say "Let me check the latest info for you..." and call the relevant tool.

                Only give helpful, up-to-date and actionable guidance.

                If the user query is vague, ask a clarifying question to narrow it down.

             """),
             ("user", "command: {command}")
        ]
    )
    llm_with_tools = llm.bind_tools(tools)
    chain = prompt|llm_with_tools
    # ai_message = chain.invoke({"command": state["messages"][-1]})
    # return {"messages": state["messages"] + [ai_message]}
    return {"messages": [chain.invoke(state["messages"])]}

# graph
def guidance(query):
    graph_builder = StateGraph(State)
    graph_builder.add_node("llm_node",tool_calling_llm)
    graph_builder.add_node("tool_node",ToolNode(tools))

    # edges
    graph_builder.add_edge(START,"llm_node")
    graph_builder.add_conditional_edges(
        "llm_node",
        tools_condition,
        {
            "tools": "tool_node",
            "__end__": END
        }
    )
    graph_builder.add_edge("tool_node","llm_node")

    graph=graph_builder.compile()

    response = graph.invoke({"messages":query})
    # last_msg = response["messages"][-1]
    # return getattr(last_msg, "content", str(last_msg))
    return response["messages"][-1].content

# web_answer("any ai discoveries today?")
