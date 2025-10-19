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


# import fitz
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_groq import ChatGroq
# import tempfile
# import os
# from dotenv import load_dotenv
# import uvicorn
# import json
# import re
# import base64
# import asyncio
# from functools import lru_cache

# load_dotenv()

# # Initialize LLM with connection pooling
# llm = ChatGroq(
#     model="meta-llama/llama-4-scout-17b-16e-instruct", 
#     temperature=0.3,
#     max_retries=2,
#     timeout=60
# )

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Rate limiting: simple in-memory tracker
# request_tracker = {}

# async def check_rate_limit(client_ip: str):
#     """Simple rate limiting: 3 requests per minute"""
#     import time
#     current_time = time.time()
    
#     if client_ip in request_tracker:
#         requests = request_tracker[client_ip]
#         # Remove old requests (older than 1 minute)
#         requests = [t for t in requests if current_time - t < 60]
        
#         if len(requests) >= 3:
#             raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 3 requests per minute.")
        
#         requests.append(current_time)
#         request_tracker[client_ip] = requests
#     else:
#         request_tracker[client_ip] = [current_time]

# def pdf_to_images(pdf_path, max_pages=3):
#     """Convert PDF to base64 images - LIMIT TO 3 PAGES for speed"""
#     doc = fitz.open(pdf_path)
#     images = []
    
#     # Only process first 3 pages (most important)
#     pages_to_process = min(len(doc), max_pages)
    
#     for i in range(pages_to_process):
#         page = doc[i]
#         # REDUCED DPI: 100 instead of 150 (saves 55% processing time)
#         pix = page.get_pixmap(matrix=fitz.Matrix(100/72, 100/72))
#         img_data = pix.tobytes("png")
#         img_base64 = base64.b64encode(img_data).decode('utf-8')
#         images.append(img_base64)
    
#     doc.close()
#     return images

# def extract_text(pdf_path, max_chars=3000):
#     """Extract text from PDF - LIMIT OUTPUT"""
#     doc = fitz.open(pdf_path)
#     text = ""
    
#     # Only extract from first 2 pages
#     for i in range(min(2, len(doc))):
#         text += doc[i].get_text()
#         if len(text) > max_chars:
#             break
    
#     doc.close()
#     return text[:max_chars]

# def clean_json_response(response_text):
#     """Clean and extract JSON from LLM response"""
#     cleaned = response_text.strip()
#     cleaned = re.sub(r'^```(?:json)?\s*\n', '', cleaned)
#     cleaned = re.sub(r'\n```\s*$', '', cleaned)
    
#     match = re.search(r'\{.*\}', cleaned, re.DOTALL)
#     if match:
#         return match.group(0)
#     return cleaned

# def parse_json_safe(json_str):
#     """Parse JSON with error handling"""
#     try:
#         return json.loads(json_str)
#     except json.JSONDecodeError as e:
#         print(f"JSON Error: {e.msg}")
#         # Try to fix
#         fixed = json_str.replace('\n', ' ').replace('\r', '')
#         try:
#             return json.loads(fixed)
#         except:
#             return None

# @lru_cache(maxsize=1)
# def get_system_prompt():
#     """Cached system prompt"""
#     return """You are a resume analyzer. Analyze quickly and provide scores.

# CRITICAL: Keep responses CONCISE. Max 80 characters per text field.

# Return ONLY valid JSON (no markdown):
# {
#     "target_role": "role",
#     "overall_score": 0,
#     "brief_overview": "short overview",
#     "ats_score": {
#         "score": 0,
#         "feedback": "brief",
#         "issues": ["issue1"],
#         "improvements": ["fix1"],
#         "missing_keywords": ["word1"]
#     },
#     "content_score": {
#         "score": 0,
#         "feedback": "brief",
#         "issues": ["issue1"],
#         "improvements": ["fix1"],
#         "strong_sections": ["section1"],
#         "weak_sections": ["section1"]
#     },
#     "structure_score": {
#         "score": 0,
#         "feedback": "brief",
#         "issues": ["issue1"],
#         "improvements": ["fix1"],
#         "missing_sections": ["section1"]
#     },
#     "skills_score": {
#         "score": 0,
#         "feedback": "brief",
#         "issues": ["issue1"],
#         "improvements": ["fix1"],
#         "relevant_skills": ["skill1"],
#         "missing_skills": ["skill1"]
#     },
#     "tone_style_score": {
#         "score": 0,
#         "feedback": "brief",
#         "issues": ["issue1"],
#         "improvements": ["fix1"]
#     },
#     "role_specific_analysis": {
#         "match_percentage": 0,
#         "key_requirements_met": ["req1"],
#         "key_requirements_missing": ["req1"],
#         "competitive_advantages": ["adv1"],
#         "red_flags": ["flag1"],
#         "priority_changes": ["change1", "change2"]
#     },
#     "career_opportunities": ["role1", "role2"],
#     "action_plan": [
#         {"priority": "High", "action": "action", "section": "Experience"}
#     ]
# }"""

# async def analyze_resume_llm(images, text, target_role):
#     """Send resume to LLM - optimized"""
    
#     content = [
#         {"type": "text", "text": f"Analyze for: {target_role}\nBe concise."}
#     ]
    
#     # Only send images (no duplicate text)
#     for img in images[:2]:  # Max 2 images
#         content.append({
#             "type": "image_url",
#             "image_url": {"url": f"data:image/png;base64,{img}"}
#         })
    
#     messages = [
#         SystemMessage(content=get_system_prompt()),
#         HumanMessage(content=content)
#     ]
    
#     # Timeout protection
#     try:
#         response = await asyncio.wait_for(
#             llm.ainvoke(messages),
#             timeout=45  # 45 second timeout
#         )
#         return response.content
#     except asyncio.TimeoutError:
#         raise HTTPException(status_code=504, detail="LLM timeout")

# def create_error_response(error_msg):
#     """Minimal error response"""
#     return {
#         "target_role": "Unknown",
#         "overall_score": 0,
#         "brief_overview": f"Error: {error_msg}",
#         "ats_score": {"score": 0, "feedback": "Error", "issues": [], "improvements": [], "missing_keywords": []},
#         "content_score": {"score": 0, "feedback": "Error", "issues": [], "improvements": [], "strong_sections": [], "weak_sections": []},
#         "structure_score": {"score": 0, "feedback": "Error", "issues": [], "improvements": [], "missing_sections": []},
#         "skills_score": {"score": 0, "feedback": "Error", "issues": [], "improvements": [], "relevant_skills": [], "missing_skills": []},
#         "tone_style_score": {"score": 0, "feedback": "Error", "issues": [], "improvements": []},
#         "role_specific_analysis": {"match_percentage": 0, "key_requirements_met": [], "key_requirements_missing": [], "competitive_advantages": [], "red_flags": [], "priority_changes": []},
#         "career_opportunities": [],
#         "action_plan": []
#     }

# @app.get("/")
# async def root():
#     return {"message": "Resume Analyzer API", "status": "running"}

# @app.get("/health")
# async def health():
#     """Health check for monitoring"""
#     return {"status": "healthy", "workers": 2}

# @app.post("/analyze-resume")
# async def analyze_resume(
#     file: UploadFile = File(...),
#     target_role: str = Form(...)
# ):
#     # Rate limiting (basic)
#     # await check_rate_limit(request.client.host)  # Uncomment if needed
    
#     # Validate
#     if not file.filename.endswith('.pdf'):
#         raise HTTPException(status_code=400, detail="PDF files only")
    
#     if not target_role.strip():
#         raise HTTPException(status_code=400, detail="Target role required")
    
#     # Check file size (limit to 5MB)
#     content = await file.read()
#     if len(content) > 5 * 1024 * 1024:
#         raise HTTPException(status_code=400, detail="File too large (max 5MB)")
    
#     # Save file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(content)
#         tmp_path = tmp.name
    
#     try:
#         # Process PDF (optimized)
#         images = pdf_to_images(tmp_path, max_pages=3)
#         text = extract_text(tmp_path, max_chars=3000)
        
#         if not images:
#             raise HTTPException(status_code=400, detail="Could not process PDF")
        
#         # Single attempt (no retry for speed)
#         print(f"Analyzing resume for: {target_role}")
        
#         llm_response = await analyze_resume_llm(images, text, target_role.strip())
        
#         # Parse
#         json_str = clean_json_response(llm_response)
#         result = parse_json_safe(json_str)
        
#         if not result or "overall_score" not in result:
#             result = create_error_response("Invalid response")
#         else:
#             # Recalculate score
#             scores = [
#                 result.get("ats_score", {}).get("score", 0),
#                 result.get("content_score", {}).get("score", 0),
#                 result.get("structure_score", {}).get("score", 0),
#                 result.get("skills_score", {}).get("score", 0),
#                 result.get("tone_style_score", {}).get("score", 0)
#             ]
#             result["overall_score"] = round(sum(scores) / 5, 1)
        
#         return {
#             "success": True,
#             "target_role": result.get("target_role", target_role),
#             "overall_score": result.get("overall_score", 0),
#             "brief_overview": result.get("brief_overview", ""),
#             "scores": {
#                 "ats": result.get("ats_score", {}),
#                 "content": result.get("content_score", {}),
#                 "structure": result.get("structure_score", {}),
#                 "skills": result.get("skills_score", {}),
#                 "tone_style": result.get("tone_style_score", {})
#             },
#             "role_analysis": result.get("role_specific_analysis", {}),
#             "career_opportunities": result.get("career_opportunities", []),
#             "action_plan": result.get("action_plan", [])
#         }
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"Error: {e}")
#         raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)

import fitz
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import tempfile
import os
from dotenv import load_dotenv
import uvicorn
import json
import re
import base64

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3)
app = FastAPI()

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
        
        return {
            "success": True,
            "data": {
                "overallScore": calculate_overall_score(scores),
                "ATS": {"score": scores["ats"], "tips": validate_tips(result.get("ats", {}).get("tips"))},
                "toneAndStyle": {"score": scores["toneAndStyle"], "tips": validate_tips(result.get("toneAndStyle", {}).get("tips"))},
                "content": {"score": scores["content"], "tips": validate_tips(result.get("content", {}).get("tips"))},
                "structure": {"score": scores["structure"], "tips": validate_tips(result.get("structure", {}).get("tips"))},
                "skills": {"score": scores["skills"], "tips": validate_tips(result.get("skills", {}).get("tips"))}
            }
        }
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))