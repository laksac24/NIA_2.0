# # import fitz
# # from typing import Annotated
# # from fastapi.middleware.cors import CORSMiddleware
# # from typing_extensions import TypedDict
# # from langgraph.graph.message import add_messages
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_groq import ChatGroq
# # from langgraph.graph import StateGraph, START, END
# # from fastapi import FastAPI, UploadFile, File
# # import tempfile
# # import os
# # from dotenv import load_dotenv
# # import uvicorn

# # load_dotenv()

# # # LangGraph State
# # class State(TypedDict):
# #     messages: Annotated[list, add_messages]

# # llm = ChatGroq(model="llama-3.3-70b-versatile")

# # def extract_text(pdf_path):
# #     doc = fitz.open(pdf_path)
# #     all_text = ""
# #     for page in doc:
# #         all_text += page.get_text()
# #     doc.close()
# #     return all_text

# # def resume_llm(state: State):
# #     prompt = ChatPromptTemplate.from_messages(
# #         [
# #             ("system", """
# #                 You are an expert career mentor and advisor for students.
# #                 Your job is to analyze the given extracted text from the user resume, and tell them:
# #                 - A brief overview of their resume.
# #                 - Which are their strong skills and which are their weak skills and where they have to focus on.
# #                 - Future career opportunities and job roles.
# #                 - How can they improve their resume.
# #                 THE RESPONSE MUST BE HUMAN LIKE IN A VERY FRIENDLY TONE.

# #                 Guidelines:
# #                 - Don't use ** or markdown formatting
# #                 - Keep the response short but well-structured and easy to read
# #                 - Use points where possible
# #                 - Be friendly but professional
# #             """),
# #             ("user", "command: {command}")
# #         ]
# #     )
# #     chain = prompt | llm
# #     return {"messages": [chain.invoke(state["messages"])]}

# # def guidance(query):
# #     graph_builder = StateGraph(State)
# #     graph_builder.add_node("resume_node", resume_llm)
# #     graph_builder.add_edge(START, "resume_node")
# #     graph_builder.add_edge("resume_node", END)
# #     graph = graph_builder.compile()
# #     response = graph.invoke({"messages": query})
# #     return response["messages"][-1].content

# # # FastAPI app
# # app = FastAPI()

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # allow all origins, change to specific domains in production
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # @app.get("/")
# # async def root():
# #     return {"message": "Resume Analyzer API is running!"}

# # @app.post("/analyze-resume")
# # async def analyze_resume(file: UploadFile = File(...)):
# #     # Save the uploaded file to temp
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
# #         tmp.write(await file.read())
# #         tmp_path = tmp.name

# #     text = extract_text(tmp_path)
# #     os.remove(tmp_path)
# #     result = guidance(text)
# #     return {"analysis": result}

# # if __name__ == "__main__":
# #     port = int(os.environ.get("PORT", 8000))
# #     uvicorn.run(app, host="0.0.0.0", port=port)


import fitz
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import tempfile
import os
from dotenv import load_dotenv
import uvicorn
import redis.asyncio as redis
import hashlib
import json
import re
import base64

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3)
app = FastAPI()

REDIS_URL = os.getenv("REDIS_CACHE_URL")
CACHE_TTL = 600

@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection on startup"""
    global redis_client
    try:
        redis_client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True
        )
        await redis_client.ping()
        print("✅ Redis connected successfully")
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        print("App will run without caching")
        redis_client = None

@app.on_event("shutdown")
async def shutdown_event():
    """Close Redis connection on shutdown"""
    global redis_client
    if redis_client:
        await redis_client.close()
        print("Redis connection closed")

def generate_cache_key(file_hash: str, target_role: str) -> str:
    """Generate cache key from file content hash and target role"""
    role_normalized = target_role.lower().strip()
    combined = f"{file_hash}:{role_normalized}"
    return f"resume:analysis:{hashlib.sha256(combined.encode()).hexdigest()}"

async def get_from_cache(cache_key: str):
    """Retrieve analysis from cache"""
    if not redis_client:
        return None
    
    try:
        cached = await redis_client.get(cache_key)
        if cached:
            print(f"✅ Cache HIT: {cache_key[:20]}...")
            return json.loads(cached)
    except Exception as e:
        print(f"Cache read error: {e}")
    
    return None

async def save_to_cache(cache_key: str, data: dict):
    """Save analysis to cache"""
    if not redis_client:
        return
    
    try:
        await redis_client.setex(
            cache_key,
            CACHE_TTL,
            json.dumps(data)
        )
        print(f"✅ Cache SAVED: {cache_key[:20]}...")
    except Exception as e:
        print(f"Cache write error: {e}")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for i in range(min(len(doc), 2)):
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(100/72, 100/72))
        images.append(base64.b64encode(pix.tobytes("png")).decode('utf-8'))
    doc.close()
    return images

SYSTEM_PROMPT = """You are an expert resume analyzer. Analyze the resume SPECIFICALLY for the target role provided.

CRITICAL: All feedback MUST be role-specific. Compare resume against role requirements.

Scoring (0-100):
- ATS: Keywords, formatting for the TARGET ROLE
- Content: Relevant experience, achievements for TARGET ROLE
- Skills: Technical/soft skills needed for TARGET ROLE
- Structure: Layout effectiveness for TARGET ROLE
- Tone: Professional style matching TARGET ROLE industry
- Return single or multiple tips whichever you think is good.

Return ONLY valid JSON:
{
    "ats": {"score": 75, "tips": [{"type": "improve", "tip": "Missing key role keywords: [specific keywords]"}]},
    "toneAndStyle": {"score": 80, "tips": [{"type": "good", "tip": "Tone matches [role] industry standards"}]},
    "content": {"score": 70, "tips": [{"type": "improve", "tip": "Add [specific role] project examples"}]},
    "structure": {"score": 85, "tips": [{"type": "improve", "tip": "Highlight [role-relevant] skills at top"}]},
    "skills": {"score": 65, "tips": [{"type": "improve", "tip": "Add [specific tools/tech for role]"}]}
}"""

def validate_score(score):
    try:
        return max(0, min(100, int(score)))
    except:
        return 70

def validate_tips(tips):
    valid = []
    for tip in (tips or [])[:5]:
        if isinstance(tip, dict) and tip.get("type") in ["good", "improve"] and tip.get("tip"):
            valid.append({"type": tip["type"], "tip": tip["tip"][:200]})
    return valid or [{"type": "improve", "tip": "Needs improvement"}]

def calculate_overall_score(scores):
    weights = {"ats": 0.25, "content": 0.25, "skills": 0.20, "structure": 0.15, "toneAndStyle": 0.15}
    return round(sum(scores.get(k, 70) * v for k, v in weights.items()))

@app.post("/analyze-resume")
async def analyze_resume(file: UploadFile = File(...), target_role: str = Form(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "PDF only")
    
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(400, "Max 5MB")
    
    file_hash = hashlib.sha256(content).hexdigest()
    cache_key = generate_cache_key(file_hash, target_role)
    
    # Try to get from cache
    cached_result = await get_from_cache(cache_key)
    if cached_result:
        return {
            "success": True,
            "cached": True,
            "data": cached_result
        }
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        images = pdf_to_images(tmp_path)
        if not images:
            raise HTTPException(400, "Invalid PDF")
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": f"TARGET ROLE: {target_role}\n\nAnalyze how well this resume matches this SPECIFIC role. Every tip must reference the role. Focus on role-specific keywords, skills, and experience."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images[0]}"}}
            ])
        ]
        
        response = await llm.ainvoke(messages)
        json_str = re.sub(r'```(?:json)?\s*|\s*```', '', response.content.strip())
        match = re.search(r'\{.*\}', json_str, re.DOTALL)
        result = json.loads(match.group(0) if match else json_str)
        
        scores = {k: validate_score(result.get(k, {}).get("score")) for k in ["ats", "toneAndStyle", "content", "structure", "skills"]}
        analysis_data = {
            "overallScore": calculate_overall_score(scores),
            "ATS": {
                "score": scores["ats"],
                "tips": validate_tips(result.get("ats", {}).get("tips"))
            },
            "toneAndStyle": {
                "score": scores["toneAndStyle"],
                "tips": validate_tips(result.get("toneAndStyle", {}).get("tips"))
            },
            "content": {
                "score": scores["content"],
                "tips": validate_tips(result.get("content", {}).get("tips"))
            },
            "structure": {
                "score": scores["structure"],
                "tips": validate_tips(result.get("structure", {}).get("tips"))
            },
            "skills": {
                "score": scores["skills"],
                "tips": validate_tips(result.get("skills", {}).get("tips"))
            }
        }
        await save_to_cache(cache_key, analysis_data)
        return {
            "success": True,
            "cached": False,
            "data": analysis_data
        }
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))