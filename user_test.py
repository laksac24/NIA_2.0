def user_test():
    print("hello")

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ======= Define State =======
class TestState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    level: str
    questions: list
    answers: list
    current_q: int

# ======= Setup =======
llm = ChatGroq(temperature=0.7, model="llama-3.3-70b-versatile")
memory = MemorySaver()

# ======= Nodes =======

# 1. Greet and detect intent
def greet_node(state: TestState) -> TestState:
    print("Bot: Hi! Would you like to take a test?")
    return state

# Ask for topic
def ask_topic_node(state: TestState) -> TestState:
    topic = input("Bot: On which topic you want to give the test?\nYou: ")
    state["topic"] = topic.strip().lower()
    state["level"] = ""
    state["questions"] = []
    state["answers"] = []
    state["current_q"] = 0
    return state

# 2. Ask for level
def ask_level_node(state: TestState) -> TestState:
    level = input("Bot: What is your level? (beginner / intermediate / advanced)\nYou: ")
    state["level"] = level.strip().lower()
    state["questions"] = []
    state["answers"] = []
    state["current_q"] = 0
    return state

# 3. Generate MCQ using LLM
def generate_question_node(state: TestState) -> TestState:
    prompt = f"""You are a test-making bot. Generate ONE multiple choice question on the topic {state["topic"]} for a user at {state['level']} level.
Return it in this format:
Question: ...
Options: 
    A. ...
    B. ...
    C. ...
    D. ...

MUST ensure that the question is not already there in {state['questions']}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state["questions"].append(response.content)
    print(f"\nBot: {response.content}")
    return state

# 4. Get Answer
def answer_node(state: TestState) -> TestState:
    ans = input(f"\nYour answer to Q{state['current_q']+1} (e.g. A/B/C/D): ").strip().upper()
    state["answers"].append(ans)
    state["current_q"] += 1
    return state

# 5. Evaluate Test
def evaluate_node(state: TestState) -> TestState:
    qa_pairs = list(zip(state["questions"], state["answers"]))
    prompt = f"""Evaluate this test performance.
For each question, show if the answer was correct (use the embedded correct answer).

Questions and user answers:
{qa_pairs}

Provide:
- Score out of {len(qa_pairs)}
- Feedback
- Suggestions to improve
"""
    result = llm.invoke([HumanMessage(content=prompt)])
    print("\nBot (Analysis):", result.content)
    return state

# ======= Conditions for Flow =======
def should_continue(state: TestState) -> str:
    if state["current_q"] >= 5:  # change number of questions here
        return "evaluate"  # Changed from "end" to "evaluate"
    return "next_question"

# ======= Build Graph =======
workflow = StateGraph(TestState)

workflow.add_node("greet", greet_node)
workflow.add_node("ask_topic", ask_topic_node)
workflow.add_node("ask_level", ask_level_node)
workflow.add_node("generate_question", generate_question_node)
workflow.add_node("get_answer", answer_node)
workflow.add_node("evaluate", evaluate_node)

# Edges
workflow.add_edge(START, "greet")
workflow.add_edge("greet", "ask_topic")
workflow.add_edge("ask_topic", "ask_level")
workflow.add_edge("ask_level", "generate_question")
workflow.add_edge("generate_question", "get_answer")

# Use conditional edges for the branching logic
workflow.add_conditional_edges("get_answer", should_continue, {
    "next_question": "generate_question",
    "evaluate": "evaluate"
})

workflow.add_edge("evaluate", END)

# Compile
app = workflow.compile(checkpointer=memory)

# ======= Run =======
if __name__ == "__main__":
    initial_state = {
        "messages": [],
        "topic" : "",
        "level": "",
        "questions": [],
        "answers": [],
        "current_q": 0
    }
    config = {"configurable": {"thread_id": "quiz_session_1"}}
    app.invoke(initial_state, config=config)