
# import fitz
# from typing import Annotated
# from typing_extensions import TypedDict
# from langgraph.graph.message import add_messages
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langgraph.graph import StateGraph, START, END
# from fastapi import FastAPI, UploadFile, File
# import tempfile
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # LangGraph State
# class State(TypedDict):
#     messages: Annotated[list, add_messages]

# llm = ChatGroq(model="Llama3-70b-8192")

# def extract_text(pdf_path):
#     doc = fitz.open(pdf_path)
#     all_text = ""
#     for page in doc:
#         all_text += page.get_text()
#     doc.close()
#     return all_text

# def resume_llm(state: State):
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", """
#                 You are an expert career mentor and advisor for students.
#                 Your job is to analyze the given extracted text from the user resume, and tell them:
#                 - A brief overview of their resume.
#                 - Which are their strong skills and which are their weak skills and where they have to focus on.
#                 - Future career opportunities and job roles.
#                 - How can they improve their resume.
#                 THE RESPONSE MUST BE HUMAN LIKE IN A VERY FRIENDLY TONE.

#                 Guidelines:
#                 - Don't use ** or markdown formatting
#                 - Keep the response short but well-structured and easy to read
#                 - Use points where possible
#                 - Be friendly but professional
#             """),
#             ("user", "command: {command}")
#         ]
#     )
#     chain = prompt | llm
#     return {"messages": [chain.invoke(state["messages"])]}

# def guidance(query):
#     graph_builder = StateGraph(State)
#     graph_builder.add_node("resume_node", resume_llm)
#     graph_builder.add_edge(START, "resume_node")
#     graph_builder.add_edge("resume_node", END)
#     graph = graph_builder.compile()
#     response = graph.invoke({"messages": query})
#     return response["messages"][-1].content

# # FastAPI app
# app = FastAPI()

# @app.post("/analyze-resume")
# async def analyze_resume(file: UploadFile = File(...)):
#     # Save the uploaded file to temp
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(await file.read())
#         tmp_path = tmp.name

#     text = extract_text(tmp_path)
#     os.remove(tmp_path)
#     result = guidance(text)
#     return {"analysis": result}
import fitz
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from fastapi import FastAPI, UploadFile, File
import tempfile
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# LangGraph State
class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(model="llama-3.3-70b-versatile")

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text()
    doc.close()
    return all_text

def resume_llm(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
                You are an expert career mentor and advisor for students.
                Your job is to analyze the given extracted text from the user resume, and tell them:
                - A brief overview of their resume.
                - Which are their strong skills and which are their weak skills and where they have to focus on.
                - Future career opportunities and job roles.
                - How can they improve their resume.
                THE RESPONSE MUST BE HUMAN LIKE IN A VERY FRIENDLY TONE.

                Guidelines:
                - Don't use ** or markdown formatting
                - Keep the response short but well-structured and easy to read
                - Use points where possible
                - Be friendly but professional
            """),
            ("user", "command: {command}")
        ]
    )
    chain = prompt | llm
    return {"messages": [chain.invoke(state["messages"])]}

def guidance(query):
    graph_builder = StateGraph(State)
    graph_builder.add_node("resume_node", resume_llm)
    graph_builder.add_edge(START, "resume_node")
    graph_builder.add_edge("resume_node", END)
    graph = graph_builder.compile()
    response = graph.invoke({"messages": query})
    return response["messages"][-1].content

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins, change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Resume Analyzer API is running!"}

@app.post("/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    # Save the uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    text = extract_text(tmp_path)
    os.remove(tmp_path)
    result = guidance(text)
    return {"analysis": result}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)