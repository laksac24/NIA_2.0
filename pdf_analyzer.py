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
# from typing import Annotated, Optional
# from fastapi.middleware.cors import CORSMiddleware
# from typing_extensions import TypedDict
# from langgraph.graph.message import add_messages
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_groq import ChatGroq
# from langgraph.graph import StateGraph, START, END
# from fastapi import FastAPI, UploadFile, File, Form
# import tempfile
# import os
# from dotenv import load_dotenv
# import uvicorn
# import json
# import re
# from pydantic import BaseModel
# import base64

# load_dotenv()

# # LangGraph State
# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#     target_role: str
#     resume_images: list
#     resume_text: str  # Keep text as fallback

# # Request model
# class ResumeAnalysisRequest(BaseModel):
#     target_role: str

# # Using Llama 4 Maverick with vision capabilities
# llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3)

# async def pdf_to_base64_images(pdf_path, dpi=150):
#     """Convert PDF pages to base64 encoded images"""
#     doc = fitz.open(pdf_path)
#     images = []
    
#     for page_num in range(len(doc)):
#         page = doc[page_num]
#         # Render page to image with specified DPI
#         pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
#         img_data = pix.tobytes("png")
        
#         # Convert to base64
#         img_base64 = base64.b64encode(img_data).decode('utf-8')
#         images.append(img_base64)
    
#     doc.close()
#     return images

# def extract_text_fallback(pdf_path):
#     """Fallback text extraction"""
#     doc = fitz.open(pdf_path)
#     all_text = ""
#     for page in doc:
#         all_text += page.get_text()
#     doc.close()
#     return all_text

# async def resume_scoring_llm(state: State):
#     system_prompt = """You are an expert career mentor and resume analyzer with 15 years of experience in technical recruitment.

# You are analyzing a VISUAL representation of a resume PDF. Pay close attention to:
# - Layout and formatting (columns, tables, headers, footers)
# - Visual hierarchy and readability
# - White space usage and text density
# - Font choices and consistency
# - Section organization and flow
# - ATS-compatibility (avoid complex layouts, graphics in text, tables)
# - Professional appearance

# Your task is to CRITICALLY and OBJECTIVELY analyze this resume for the specific target role. BE HARSH where needed - most resumes have significant room for improvement.

# CRITICAL SCORING RULES:
# 1. ACTUALLY ANALYZE the resume - look at both visual formatting AND content
# 2. Scores should VARY significantly based on actual quality
# 3. Score 60-70 = "acceptable but needs improvement"
# 4. Score 70-80 = "good with minor issues"
# 5. Score 80-90 = "excellent, professional quality"
# 6. Score 90+ = RARE, only for exceptional resumes
# 7. If something is missing or poor, REFLECT IT IN THE SCORE

# SCORING METHODOLOGY (0-100 for each category):

# **ATS Score (Applicant Tracking System Compatibility):**
# Start at 50 (baseline)
# - SUBTRACT 15-20 for complex layouts (multi-column, text boxes, headers/footers with info)
# - SUBTRACT 10-15 for each missing critical keyword for target role
# - SUBTRACT 10 for graphics, images, or charts
# - SUBTRACT 5-10 for missing standard sections
# - ADD 10-15 for clean, single-column layout
# - ADD 10 for quantifiable achievements with metrics
# - ADD 5-10 for proper section hierarchy

# **Content Score (Relevance & Impact):**
# Start at 50 (baseline)
# - SUBTRACT 20-25 if experience is NOT relevant to target role
# - SUBTRACT 15-20 for generic bullets ("responsible for", "worked on")
# - SUBTRACT 10-15 for lack of quantifiable results
# - ADD 15-20 for highly relevant experience matching role requirements
# - ADD 15 for strong achievement statements with specific metrics
# - ADD 10 if experience level matches role needs

# **Structure Score (Organization & Visual Format):**
# Start at 50 (baseline)
# - SUBTRACT 15-20 for poor visual hierarchy or cluttered layout
# - SUBTRACT 10-15 for inconsistent formatting (fonts, spacing, alignment)
# - SUBTRACT 10 if resume is too long (>2 pages for <10 years exp) or too short
# - SUBTRACT 10 for poor readability (dense text, no white space)
# - ADD 15-20 for excellent visual hierarchy and scannable layout
# - ADD 10-15 for consistent, professional formatting
# - ADD 5-10 for appropriate length and good use of white space

# **Skills Score (Technical & Soft Skills for Role):**
# Start at 50 (baseline)
# - SUBTRACT 25-35 if CRITICAL required skills for role are MISSING
# - SUBTRACT 15-20 if skills listed without demonstration in experience
# - SUBTRACT 10 for irrelevant or outdated skills prominently featured
# - ADD 20-25 if ALL required skills for role are present
# - ADD 15 for clear skills demonstration with context/examples
# - ADD 10 for excellent balance of technical and soft skills

# **Tone & Style Score (Professional Communication):**
# Start at 50 (baseline)
# - SUBTRACT 20-25 for excessive passive voice or weak verbs
# - SUBTRACT 15 for grammatical errors, typos, or inconsistencies
# - SUBTRACT 10 for overly verbose or unclear writing
# - ADD 20-25 for strong action verbs and clear impact statements
# - ADD 15 for concise, professional, error-free language
# - ADD 10 for appropriate tone for target role level

# **Overall Score = (ATS + Content + Structure + Skills + Tone) / 5**

# RESPOND WITH ONLY VALID JSON (no markdown, no code blocks):

# {
#     "target_role": "<exact target role>",
#     "overall_score": <calculated average>,
#     "brief_overview": "<2-3 honest sentences about resume quality and role fit>",
    
#     "ats_score": {
#         "score": <0-100>,
#         "feedback": "<2-3 sentences explaining score with specific visual observations>",
#         "issues": ["specific formatting issue seen in resume", "another specific issue"],
#         "improvements": ["specific fix based on what you see", "another fix"],
#         "missing_keywords": ["actual keyword needed for role", "another keyword"],
#         "role_relevance": "<honest ATS assessment for this role>"
#     },
    
#     "content_score": {
#         "score": <0-100>,
#         "feedback": "<2-3 sentences about content quality and relevance>",
#         "issues": ["specific content problem observed", "another problem"],
#         "improvements": ["add specific metric to section X", "rewrite bullet Y to show impact"],
#         "strong_sections": ["actual strong section name"],
#         "weak_sections": ["actual weak section name"],
#         "role_alignment": "<specific assessment for target role>"
#     },
    
#     "structure_score": {
#         "score": <0-100>,
#         "feedback": "<2-3 sentences about visual structure and organization>",
#         "issues": ["specific visual/structural problem", "another problem"],
#         "improvements": ["specific layout improvement", "another improvement"],
#         "missing_sections": ["section needed for role", "another section"],
#         "layout_suggestions": ["specific visual improvement", "another suggestion"]
#     },
    
#     "skills_score": {
#         "score": <0-100>,
#         "feedback": "<2-3 sentences about skills match to role>",
#         "issues": ["specific skill gap for this role", "another gap"],
#         "improvements": ["add specific skill X with context", "demonstrate skill Y"],
#         "relevant_skills": ["skill from resume relevant to role"],
#         "missing_skills": ["critical missing skill", "recommended skill"],
#         "skills_to_highlight": ["skill to emphasize", "skill to move up"],
#         "role_match": "<honest percentage or assessment>"
#     },
    
#     "tone_style_score": {
#         "score": <0-100>,
#         "feedback": "<2-3 sentences about writing quality>",
#         "issues": ["specific language issue", "another issue"],
#         "improvements": ["specific rewording suggestion", "another suggestion"],
#         "language_suggestions": ["replace passive voice in X", "strengthen verbs in Y"],
#         "professionalism_notes": "<tone assessment for role>"
#     },
    
#     "role_specific_analysis": {
#         "match_percentage": <0-100>,
#         "key_requirements_met": ["requirement actually shown", "another met"],
#         "key_requirements_missing": ["requirement not demonstrated", "another missing"],
#         "competitive_advantages": ["actual strength for role", "another strength"],
#         "red_flags": ["actual concern for recruiters", "another concern"],
#         "priority_changes": ["most critical change", "second priority", "third priority"]
#     },
    
#     "career_opportunities": ["highly aligned role", "alternative role", "growth path role"],
    
#     "action_plan": [
#         {"priority": "High", "action": "specific actionable change", "section": "section name"},
#         {"priority": "High", "action": "another critical change", "section": "section name"},
#         {"priority": "Medium", "action": "important improvement", "section": "section name"},
#         {"priority": "Low", "action": "enhancement", "section": "section name"}
#     ]
# }

# BE SPECIFIC - reference what you actually see in the resume visually and content-wise.
# BE CRITICAL - most resumes are NOT 80+ quality.
# VARY SCORES - different resumes should get different scores based on quality."""
    
#     # Build message content with images
#     content_parts = [
#         {
#             "type": "text",
#             "text": f"""Analyze this resume for the role of: {state['target_role']}

# Look carefully at the visual formatting, layout, and content. Be specific about what you observe."""
#         }
#     ]
    
#     # Add each page image
#     for idx, img_base64 in enumerate(state['resume_images']):
#         content_parts.append({
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/png;base64,{img_base64}"
#             }
#         })
    
#     # Add text as additional context if needed
#     if state.get('resume_text'):
#         content_parts.append({
#             "type": "text",
#             "text": f"\n\nResume Text Content (for reference):\n{state['resume_text'][:6000]}"
#         })
    
#     messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=content_parts)
#     ]
    
#     response = await llm.ainvoke(messages)
#     return {"messages": [response]}

# def parse_llm_response(response_text):
#     """Parse the LLM response and extract JSON"""
#     try:
#         # Remove markdown code blocks if present
#         cleaned = response_text.strip()
#         if cleaned.startswith("```"):
#             cleaned = re.sub(r'^```(?:json)?\n', '', cleaned)
#             cleaned = re.sub(r'\n```$', '', cleaned)
        
#         # Try to find JSON in the response
#         json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
#         if json_match:
#             json_str = json_match.group(0)
#             parsed = json.loads(json_str)
            
#             # Validate and recalculate overall score
#             scores = [
#                 parsed.get("ats_score", {}).get("score", 0),
#                 parsed.get("content_score", {}).get("score", 0),
#                 parsed.get("structure_score", {}).get("score", 0),
#                 parsed.get("skills_score", {}).get("score", 0),
#                 parsed.get("tone_style_score", {}).get("score", 0)
#             ]
            
#             if all(isinstance(s, (int, float)) for s in scores):
#                 calculated_overall = round(sum(scores) / 5, 1)
#                 parsed["overall_score"] = calculated_overall
            
#             return parsed
#         else:
#             return json.loads(cleaned)
#     except json.JSONDecodeError as e:
#         print(f"JSON Parse Error: {e}")
#         print(f"Response: {response_text[:500]}")
#         return {
#             "target_role": "Unknown",
#             "overall_score": 0,
#             "brief_overview": "Error parsing resume analysis. Please try again.",
#             "ats_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "missing_keywords": [], "role_relevance": "Unable to determine"},
#             "content_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "strong_sections": [], "weak_sections": [], "role_alignment": "Unable to determine"},
#             "structure_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "missing_sections": [], "layout_suggestions": []},
#             "skills_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "relevant_skills": [], "missing_skills": [], "skills_to_highlight": [], "role_match": "Unable to determine"},
#             "tone_style_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "language_suggestions": [], "professionalism_notes": "Unable to determine"},
#             "role_specific_analysis": {"match_percentage": 0, "key_requirements_met": [], "key_requirements_missing": [], "competitive_advantages": [], "red_flags": [], "priority_changes": []},
#             "career_opportunities": [],
#             "action_plan": []
#         }

# async def analyze_resume_with_scores(resume_images, resume_text, target_role):
#     graph_builder = StateGraph(State)
#     graph_builder.add_node("scoring_node", resume_scoring_llm)
#     graph_builder.add_edge(START, "scoring_node")
#     graph_builder.add_edge("scoring_node", END)
#     graph = graph_builder.compile()
    
#     response = await graph.ainvoke({
#         "messages": [],
#         "target_role": target_role,
#         "resume_images": resume_images,
#         "resume_text": resume_text
#     })
    
#     # Parse the response
#     llm_output = response["messages"][-1].content
#     parsed_result = parse_llm_response(llm_output)
    
#     return parsed_result

# # FastAPI app
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# async def root():
#     return {"message": "Resume Analyzer API with Vision-Based Scoring is running!"}

# @app.post("/analyze-resume")
# async def analyze_resume(
#     file: UploadFile = File(...),
#     target_role: str = Form(...)
# ):
#     # Validate inputs
#     if not file.filename.endswith('.pdf'):
#         return {"success": False, "error": "Only PDF files are supported"}
    
#     if not target_role or len(target_role.strip()) == 0:
#         return {"success": False, "error": "Target role is required"}
    
#     # Save the uploaded file to temp
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(await file.read())
#         tmp_path = tmp.name

#     try:
#         # Convert PDF to images for vision analysis
#         resume_images = pdf_to_base64_images(tmp_path, dpi=150)
        
#         # Also extract text as fallback/additional context
#         resume_text = extract_text_fallback(tmp_path)
        
#         if not resume_images:
#             return {"success": False, "error": "Could not process PDF pages"}
        
#         if not resume_text.strip():
#             return {"success": False, "error": "No text could be extracted from the PDF"}
        
#         result = analyze_resume_with_scores(resume_images, resume_text, target_role.strip())
        
#         return {
#             "success": True,
#             "target_role": result["target_role"],
#             "overall_score": result["overall_score"],
#             "brief_overview": result["brief_overview"],
            
#             "scores": {
#                 "ats": result["ats_score"],
#                 "content": result["content_score"],
#                 "structure": result["structure_score"],
#                 "skills": result["skills_score"],
#                 "tone_style": result["tone_style_score"]
#             },
            
#             "role_analysis": result["role_specific_analysis"],
#             "career_opportunities": result["career_opportunities"],
#             "action_plan": result["action_plan"]
#         }
    
#     except Exception as e:
#         import traceback
#         return {"success": False, "error": f"Analysis failed: {str(e)}", "traceback": traceback.format_exc()}
    
#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)




import fitz
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
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
import asyncio
from functools import lru_cache

load_dotenv()

# Initialize LLM with connection pooling
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", 
    temperature=0.3,
    max_retries=2,
    timeout=60
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting: simple in-memory tracker
request_tracker = {}

async def check_rate_limit(client_ip: str):
    """Simple rate limiting: 3 requests per minute"""
    import time
    current_time = time.time()
    
    if client_ip in request_tracker:
        requests = request_tracker[client_ip]
        # Remove old requests (older than 1 minute)
        requests = [t for t in requests if current_time - t < 60]
        
        if len(requests) >= 3:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 3 requests per minute.")
        
        requests.append(current_time)
        request_tracker[client_ip] = requests
    else:
        request_tracker[client_ip] = [current_time]

def pdf_to_images(pdf_path, max_pages=3):
    """Convert PDF to base64 images - LIMIT TO 3 PAGES for speed"""
    doc = fitz.open(pdf_path)
    images = []
    
    # Only process first 3 pages (most important)
    pages_to_process = min(len(doc), max_pages)
    
    for i in range(pages_to_process):
        page = doc[i]
        # REDUCED DPI: 100 instead of 150 (saves 55% processing time)
        pix = page.get_pixmap(matrix=fitz.Matrix(100/72, 100/72))
        img_data = pix.tobytes("png")
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        images.append(img_base64)
    
    doc.close()
    return images

def extract_text(pdf_path, max_chars=3000):
    """Extract text from PDF - LIMIT OUTPUT"""
    doc = fitz.open(pdf_path)
    text = ""
    
    # Only extract from first 2 pages
    for i in range(min(2, len(doc))):
        text += doc[i].get_text()
        if len(text) > max_chars:
            break
    
    doc.close()
    return text[:max_chars]

def clean_json_response(response_text):
    """Clean and extract JSON from LLM response"""
    cleaned = response_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*\n', '', cleaned)
    cleaned = re.sub(r'\n```\s*$', '', cleaned)
    
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        return match.group(0)
    return cleaned

def parse_json_safe(json_str):
    """Parse JSON with error handling"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e.msg}")
        # Try to fix
        fixed = json_str.replace('\n', ' ').replace('\r', '')
        try:
            return json.loads(fixed)
        except:
            return None

@lru_cache(maxsize=1)
def get_system_prompt():
    """Cached system prompt"""
    return """You are a resume analyzer. Analyze quickly and provide scores.

CRITICAL: Keep responses CONCISE. Max 80 characters per text field.

Return ONLY valid JSON (no markdown):
{
    "target_role": "role",
    "overall_score": 0,
    "brief_overview": "short overview",
    "ats_score": {
        "score": 0,
        "feedback": "brief",
        "issues": ["issue1"],
        "improvements": ["fix1"],
        "missing_keywords": ["word1"]
    },
    "content_score": {
        "score": 0,
        "feedback": "brief",
        "issues": ["issue1"],
        "improvements": ["fix1"],
        "strong_sections": ["section1"],
        "weak_sections": ["section1"]
    },
    "structure_score": {
        "score": 0,
        "feedback": "brief",
        "issues": ["issue1"],
        "improvements": ["fix1"],
        "missing_sections": ["section1"]
    },
    "skills_score": {
        "score": 0,
        "feedback": "brief",
        "issues": ["issue1"],
        "improvements": ["fix1"],
        "relevant_skills": ["skill1"],
        "missing_skills": ["skill1"]
    },
    "tone_style_score": {
        "score": 0,
        "feedback": "brief",
        "issues": ["issue1"],
        "improvements": ["fix1"]
    },
    "role_specific_analysis": {
        "match_percentage": 0,
        "key_requirements_met": ["req1"],
        "key_requirements_missing": ["req1"],
        "competitive_advantages": ["adv1"],
        "red_flags": ["flag1"],
        "priority_changes": ["change1", "change2"]
    },
    "career_opportunities": ["role1", "role2"],
    "action_plan": [
        {"priority": "High", "action": "action", "section": "Experience"}
    ]
}"""

async def analyze_resume_llm(images, text, target_role):
    """Send resume to LLM - optimized"""
    
    content = [
        {"type": "text", "text": f"Analyze for: {target_role}\nBe concise."}
    ]
    
    # Only send images (no duplicate text)
    for img in images[:2]:  # Max 2 images
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img}"}
        })
    
    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=content)
    ]
    
    # Timeout protection
    try:
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=45  # 45 second timeout
        )
        return response.content
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="LLM timeout")

def create_error_response(error_msg):
    """Minimal error response"""
    return {
        "target_role": "Unknown",
        "overall_score": 0,
        "brief_overview": f"Error: {error_msg}",
        "ats_score": {"score": 0, "feedback": "Error", "issues": [], "improvements": [], "missing_keywords": []},
        "content_score": {"score": 0, "feedback": "Error", "issues": [], "improvements": [], "strong_sections": [], "weak_sections": []},
        "structure_score": {"score": 0, "feedback": "Error", "issues": [], "improvements": [], "missing_sections": []},
        "skills_score": {"score": 0, "feedback": "Error", "issues": [], "improvements": [], "relevant_skills": [], "missing_skills": []},
        "tone_style_score": {"score": 0, "feedback": "Error", "issues": [], "improvements": []},
        "role_specific_analysis": {"match_percentage": 0, "key_requirements_met": [], "key_requirements_missing": [], "competitive_advantages": [], "red_flags": [], "priority_changes": []},
        "career_opportunities": [],
        "action_plan": []
    }

@app.get("/")
async def root():
    return {"message": "Resume Analyzer API", "status": "running"}

@app.get("/health")
async def health():
    """Health check for monitoring"""
    return {"status": "healthy", "workers": 2}

@app.post("/analyze-resume")
async def analyze_resume(
    file: UploadFile = File(...),
    target_role: str = Form(...)
):
    # Rate limiting (basic)
    # await check_rate_limit(request.client.host)  # Uncomment if needed
    
    # Validate
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF files only")
    
    if not target_role.strip():
        raise HTTPException(status_code=400, detail="Target role required")
    
    # Check file size (limit to 5MB)
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 5MB)")
    
    # Save file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Process PDF (optimized)
        images = pdf_to_images(tmp_path, max_pages=3)
        text = extract_text(tmp_path, max_chars=3000)
        
        if not images:
            raise HTTPException(status_code=400, detail="Could not process PDF")
        
        # Single attempt (no retry for speed)
        print(f"Analyzing resume for: {target_role}")
        
        llm_response = await analyze_resume_llm(images, text, target_role.strip())
        
        # Parse
        json_str = clean_json_response(llm_response)
        result = parse_json_safe(json_str)
        
        if not result or "overall_score" not in result:
            result = create_error_response("Invalid response")
        else:
            # Recalculate score
            scores = [
                result.get("ats_score", {}).get("score", 0),
                result.get("content_score", {}).get("score", 0),
                result.get("structure_score", {}).get("score", 0),
                result.get("skills_score", {}).get("score", 0),
                result.get("tone_style_score", {}).get("score", 0)
            ]
            result["overall_score"] = round(sum(scores) / 5, 1)
        
        return {
            "success": True,
            "target_role": result.get("target_role", target_role),
            "overall_score": result.get("overall_score", 0),
            "brief_overview": result.get("brief_overview", ""),
            "scores": {
                "ats": result.get("ats_score", {}),
                "content": result.get("content_score", {}),
                "structure": result.get("structure_score", {}),
                "skills": result.get("skills_score", {}),
                "tone_style": result.get("tone_style_score", {})
            },
            "role_analysis": result.get("role_specific_analysis", {}),
            "career_opportunities": result.get("career_opportunities", []),
            "action_plan": result.get("action_plan", [])
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)