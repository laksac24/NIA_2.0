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

# # import fitz
# # from typing import Annotated, Optional
# # from fastapi.middleware.cors import CORSMiddleware
# # from typing_extensions import TypedDict
# # from langgraph.graph.message import add_messages
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_groq import ChatGroq
# # from langgraph.graph import StateGraph, START, END
# # from fastapi import FastAPI, UploadFile, File, Form
# # import tempfile
# # import os
# # from dotenv import load_dotenv
# # import uvicorn
# # import json
# # import re
# # from pydantic import BaseModel

# # load_dotenv()

# # # LangGraph State
# # class State(TypedDict):
# #     messages: Annotated[list, add_messages]

# # # Request model
# # class ResumeAnalysisRequest(BaseModel):
# #     target_role: str

# # llm = ChatGroq(model="llama-3.3-70b-versatile")

# # def extract_text(pdf_path):
# #     doc = fitz.open(pdf_path)
# #     all_text = ""
# #     for page in doc:
# #         all_text += page.get_text()
# #     doc.close()
# #     return all_text

# # def resume_scoring_llm(state: State):
# #     prompt = ChatPromptTemplate.from_messages(
# #         [
# #             ("system", """
# #                 You are an expert career mentor and resume analyzer.
# #                 Analyze the given resume text FOR THE SPECIFIC TARGET ROLE provided and give detailed, actionable feedback.
                
# #                 You MUST respond with a valid JSON object (no markdown, no code blocks) with the following structure:
# #                 {{
# #                     "target_role": "<the role being analyzed for>",
# #                     "overall_score": <calculated average of all scores>,
# #                     "brief_overview": "<2-3 sentences about the resume relevance to the target role>",
                    
# #                     "ats_score": {{
# #                         "score": <number between 0-100>,
# #                         "feedback": "<overall ATS feedback>",
# #                         "issues": ["specific issue 1", "specific issue 2"],
# #                         "improvements": ["specific improvement 1", "specific improvement 2"],
# #                         "missing_keywords": ["keyword1", "keyword2"],
# #                         "role_relevance": "<how well optimized for the target role>"
# #                     }},
                    
# #                     "content_score": {{
# #                         "score": <number between 0-100>,
# #                         "feedback": "<overall content feedback>",
# #                         "issues": ["specific issue 1", "specific issue 2"],
# #                         "improvements": ["add quantifiable achievement in X section", "rewrite Y bullet to show impact"],
# #                         "strong_sections": ["section that works well"],
# #                         "weak_sections": ["section that needs work"],
# #                         "role_alignment": "<how content aligns with target role requirements>"
# #                     }},
                    
# #                     "structure_score": {{
# #                         "score": <number between 0-100>,
# #                         "feedback": "<overall structure feedback>",
# #                         "issues": ["specific formatting issue 1", "section X is poorly organized"],
# #                         "improvements": ["move section X before Y", "add section Z"],
# #                         "missing_sections": ["section1", "section2"],
# #                         "layout_suggestions": ["specific layout improvement 1", "specific layout improvement 2"]
# #                     }},
                    
# #                     "skills_score": {{
# #                         "score": <number between 0-100>,
# #                         "feedback": "<overall skills feedback>",
# #                         "issues": ["missing critical skill X", "skill Y not demonstrated"],
# #                         "improvements": ["add skill X with context", "provide evidence for skill Y"],
# #                         "relevant_skills": ["skill1", "skill2"],
# #                         "missing_skills": ["skill1 required for role", "skill2 recommended"],
# #                         "skills_to_highlight": ["emphasize this skill more", "move this skill up"],
# #                         "role_match": "<percentage or description of skill match to target role>"
# #                     }},
                    
# #                     "tone_style_score": {{
# #                         "score": <number between 0-100>,
# #                         "feedback": "<overall tone feedback>",
# #                         "issues": ["passive voice in section X", "weak action verbs in Y"],
# #                         "improvements": ["replace 'responsible for' with action verbs", "make bullets more concise"],
# #                         "language_suggestions": ["specific rewording suggestion 1", "specific rewording suggestion 2"],
# #                         "professionalism_notes": "<how professional it sounds for the target role>"
# #                     }},
                    
# #                     "role_specific_analysis": {{
# #                         "match_percentage": <0-100>,
# #                         "key_requirements_met": ["requirement 1", "requirement 2"],
# #                         "key_requirements_missing": ["requirement 1", "requirement 2"],
# #                         "competitive_advantages": ["what makes this resume stand out for this role"],
# #                         "red_flags": ["what might concern recruiters for this role"],
# #                         "priority_changes": ["most critical change 1", "most critical change 2", "most critical change 3"]
# #                     }},
                    
# #                     "career_opportunities": ["role 1 similar to target", "role 2 alternative", "role 3 growth path"],
                    
# #                     "action_plan": [
# #                         {{"priority": "High", "action": "specific action 1", "section": "which section to change"}},
# #                         {{"priority": "High", "action": "specific action 2", "section": "which section to change"}},
# #                         {{"priority": "Medium", "action": "specific action 3", "section": "which section to change"}},
# #                         {{"priority": "Low", "action": "specific action 4", "section": "which section to change"}}
# #                     ]
# #                 }}

# #                 SCORING CRITERIA (all scored 0-100 based on relevance to TARGET ROLE):
                
# #                 - ATS Score: 
# #                   * Keyword match for target role (30 points)
# #                   * Format compatibility (25 points)
# #                   * Proper sections for role (25 points)
# #                   * Quantifiable achievements (20 points)
                
# #                 - Content Score:
# #                   * Relevance to target role (35 points)
# #                   * Impact and achievements (30 points)
# #                   * Quantifiable results (20 points)
# #                   * Experience level match (15 points)
                
# #                 - Structure Score:
# #                   * Organization for role (30 points)
# #                   * Clarity and readability (25 points)
# #                   * Formatting consistency (25 points)
# #                   * Appropriate length (20 points)
                
# #                 - Skills Score:
# #                   * Required skills for role (40 points)
# #                   * Skill demonstration/evidence (25 points)
# #                   * Technical vs soft skills balance (20 points)
# #                   * Breadth and depth (15 points)
                
# #                 - Tone & Style Score:
# #                   * Professional tone for role (30 points)
# #                   * Action verbs and impact (25 points)
# #                   * Clarity and conciseness (25 points)
# #                   * Grammar and errors (20 points)

# #                 Overall Score: Calculate as (ATS + Content + Structure + Skills + Tone) / 5

# #                 Be SPECIFIC in all feedback. Instead of "improve your skills section", say "add Python and SQL in skills section as they're required for Data Analyst role" or "move your data analysis skills to the top of your skills list".
                
# #                 Keep friendly but professional tone. No markdown formatting.
# #             """),
# #             ("user", "Target Role: {target_role}\n\nResume text: {resume_text}")
# #         ]
# #     )
# #     chain = prompt | llm
# #     return {"messages": [chain.invoke(state["messages"])]}

# # def parse_llm_response(response_text):
# #     """Parse the LLM response and extract JSON"""
# #     try:
# #         # Try to find JSON in the response
# #         json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
# #         if json_match:
# #             json_str = json_match.group(0)
# #             return json.loads(json_str)
# #         else:
# #             return json.loads(response_text)
# #     except json.JSONDecodeError as e:
# #         print(f"JSON Parse Error: {e}")
# #         print(f"Response: {response_text[:500]}")
# #         # Fallback response
# #         return {
# #             "target_role": "Unknown",
# #             "overall_score": 0,
# #             "brief_overview": "Error parsing resume analysis. Please try again.",
# #             "ats_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "missing_keywords": [], "role_relevance": "Unable to determine"},
# #             "content_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "strong_sections": [], "weak_sections": [], "role_alignment": "Unable to determine"},
# #             "structure_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "missing_sections": [], "layout_suggestions": []},
# #             "skills_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "relevant_skills": [], "missing_skills": [], "skills_to_highlight": [], "role_match": "Unable to determine"},
# #             "tone_style_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "language_suggestions": [], "professionalism_notes": "Unable to determine"},
# #             "role_specific_analysis": {"match_percentage": 0, "key_requirements_met": [], "key_requirements_missing": [], "competitive_advantages": [], "red_flags": [], "priority_changes": []},
# #             "career_opportunities": [],
# #             "action_plan": []
# #         }

# # def analyze_resume_with_scores(resume_text, target_role):
# #     graph_builder = StateGraph(State)
# #     graph_builder.add_node("scoring_node", resume_scoring_llm)
# #     graph_builder.add_edge(START, "scoring_node")
# #     graph_builder.add_edge("scoring_node", END)
# #     graph = graph_builder.compile()
    
# #     # Format the message correctly for LangGraph
# #     user_message = f"Target Role: {target_role}\n\nResume text: {resume_text}"
    
# #     response = graph.invoke({
# #         "messages": [{"role": "user", "content": user_message}]
# #     })
    
# #     # Parse the response
# #     llm_output = response["messages"][-1].content
# #     parsed_result = parse_llm_response(llm_output)
    
# #     return parsed_result

# # # FastAPI app
# # app = FastAPI()

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # @app.get("/")
# # async def root():
# #     return {"message": "Resume Analyzer API with Role-Based Scoring is running!"}

# # @app.post("/analyze-resume")
# # async def analyze_resume(
# #     file: UploadFile = File(...),
# #     target_role: str = Form(...)
# # ):
# #     # Validate inputs
# #     if not file.filename.endswith('.pdf'):
# #         return {"success": False, "error": "Only PDF files are supported"}
    
# #     if not target_role or len(target_role.strip()) == 0:
# #         return {"success": False, "error": "Target role is required"}
    
# #     # Save the uploaded file to temp
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
# #         tmp.write(await file.read())
# #         tmp_path = tmp.name

# #     try:
# #         text = extract_text(tmp_path)
        
# #         if not text.strip():
# #             return {"success": False, "error": "No text could be extracted from the PDF"}
        
# #         result = analyze_resume_with_scores(text, target_role.strip())
        
# #         return {
# #             "success": True,
# #             "target_role": result["target_role"],
# #             "overall_score": result["overall_score"],
# #             "brief_overview": result["brief_overview"],
            
# #             "scores": {
# #                 "ats": result["ats_score"],
# #                 "content": result["content_score"],
# #                 "structure": result["structure_score"],
# #                 "skills": result["skills_score"],
# #                 "tone_style": result["tone_style_score"]
# #             },
            
# #             "role_analysis": result["role_specific_analysis"],
# #             "career_opportunities": result["career_opportunities"],
# #             "action_plan": result["action_plan"]
# #         }
    
# #     except Exception as e:
# #         return {"success": False, "error": f"Analysis failed: {str(e)}"}
    
# #     finally:
# #         if os.path.exists(tmp_path):
# #             os.remove(tmp_path)

# # if __name__ == "__main__":
# #     port = int(os.environ.get("PORT", 8000))
# #     uvicorn.run(app, host="0.0.0.0", port=port)



# # import fitz
# # from typing import Annotated, Optional
# # from fastapi.middleware.cors import CORSMiddleware
# # from typing_extensions import TypedDict
# # from langgraph.graph.message import add_messages
# # from langchain_core.messages import HumanMessage, SystemMessage
# # from langchain_groq import ChatGroq
# # from langgraph.graph import StateGraph, START, END
# # from fastapi import FastAPI, UploadFile, File, Form
# # import tempfile
# # import os
# # from dotenv import load_dotenv
# # import uvicorn
# # import json
# # import re
# # from pydantic import BaseModel

# # load_dotenv()

# # # LangGraph State
# # class State(TypedDict):
# #     messages: Annotated[list, add_messages]
# #     target_role: str
# #     resume_text: str

# # # Request model
# # class ResumeAnalysisRequest(BaseModel):
# #     target_role: str

# # llm = ChatGroq(model="llama-3.3-70b-versatile")

# # def extract_text(pdf_path):
# #     doc = fitz.open(pdf_path)
# #     all_text = ""
# #     for page in doc:
# #         all_text += page.get_text()
# #     doc.close()
# #     return all_text

# # def resume_scoring_llm(state: State):
# #     system_prompt = """
# # You are an expert career mentor and resume analyzer.
# # Analyze the given resume text FOR THE SPECIFIC TARGET ROLE provided and give detailed, actionable feedback.

# # You MUST respond with a valid JSON object (no markdown, no code blocks) with the following structure:
# # {
# #     "target_role": "<the role being analyzed for>",
# #     "overall_score": <calculated average of all scores>,
# #     "brief_overview": "<2-3 sentences about the resume relevance to the target role>",
    
# #     "ats_score": {
# #         "score": <number between 0-100>,
# #         "feedback": "<overall ATS feedback>",
# #         "issues": ["specific issue 1", "specific issue 2"],
# #         "improvements": ["specific improvement 1", "specific improvement 2"],
# #         "missing_keywords": ["keyword1", "keyword2"],
# #         "role_relevance": "<how well optimized for the target role>"
# #     },
    
# #     "content_score": {
# #         "score": <number between 0-100>,
# #         "feedback": "<overall content feedback>",
# #         "issues": ["specific issue 1", "specific issue 2"],
# #         "improvements": ["add quantifiable achievement in X section", "rewrite Y bullet to show impact"],
# #         "strong_sections": ["section that works well"],
# #         "weak_sections": ["section that needs work"],
# #         "role_alignment": "<how content aligns with target role requirements>"
# #     },
    
# #     "structure_score": {
# #         "score": <number between 0-100>,
# #         "feedback": "<overall structure feedback>",
# #         "issues": ["specific formatting issue 1", "section X is poorly organized"],
# #         "improvements": ["move section X before Y", "add section Z"],
# #         "missing_sections": ["section1", "section2"],
# #         "layout_suggestions": ["specific layout improvement 1", "specific layout improvement 2"]
# #     },
    
# #     "skills_score": {
# #         "score": <number between 0-100>,
# #         "feedback": "<overall skills feedback>",
# #         "issues": ["missing critical skill X", "skill Y not demonstrated"],
# #         "improvements": ["add skill X with context", "provide evidence for skill Y"],
# #         "relevant_skills": ["skill1", "skill2"],
# #         "missing_skills": ["skill1 required for role", "skill2 recommended"],
# #         "skills_to_highlight": ["emphasize this skill more", "move this skill up"],
# #         "role_match": "<percentage or description of skill match to target role>"
# #     },
    
# #     "tone_style_score": {
# #         "score": <number between 0-100>,
# #         "feedback": "<overall tone feedback>",
# #         "issues": ["passive voice in section X", "weak action verbs in Y"],
# #         "improvements": ["replace 'responsible for' with action verbs", "make bullets more concise"],
# #         "language_suggestions": ["specific rewording suggestion 1", "specific rewording suggestion 2"],
# #         "professionalism_notes": "<how professional it sounds for the target role>"
# #     },
    
# #     "role_specific_analysis": {
# #         "match_percentage": <0-100>,
# #         "key_requirements_met": ["requirement 1", "requirement 2"],
# #         "key_requirements_missing": ["requirement 1", "requirement 2"],
# #         "competitive_advantages": ["what makes this resume stand out for this role"],
# #         "red_flags": ["what might concern recruiters for this role"],
# #         "priority_changes": ["most critical change 1", "most critical change 2", "most critical change 3"]
# #     },
    
# #     "career_opportunities": ["role 1 similar to target", "role 2 alternative", "role 3 growth path"],
    
# #     "action_plan": [
# #         {"priority": "High", "action": "specific action 1", "section": "which section to change"},
# #         {"priority": "High", "action": "specific action 2", "section": "which section to change"},
# #         {"priority": "Medium", "action": "specific action 3", "section": "which section to change"},
# #         {"priority": "Low", "action": "specific action 4", "section": "which section to change"}
# #     ]
# # }

# # SCORING CRITERIA (all scored 0-100 based on relevance to TARGET ROLE):

# # - ATS Score: 
# #   * Keyword match for target role (30 points)
# #   * Format compatibility (25 points)
# #   * Proper sections for role (25 points)
# #   * Quantifiable achievements (20 points)

# # - Content Score:
# #   * Relevance to target role (35 points)
# #   * Impact and achievements (30 points)
# #   * Quantifiable results (20 points)
# #   * Experience level match (15 points)

# # - Structure Score:
# #   * Organization for role (30 points)
# #   * Clarity and readability (25 points)
# #   * Formatting consistency (25 points)
# #   * Appropriate length (20 points)

# # - Skills Score:
# #   * Required skills for role (40 points)
# #   * Skill demonstration/evidence (25 points)
# #   * Technical vs soft skills balance (20 points)
# #   * Breadth and depth (15 points)

# # - Tone & Style Score:
# #   * Professional tone for role (30 points)
# #   * Action verbs and impact (25 points)
# #   * Clarity and conciseness (25 points)
# #   * Grammar and errors (20 points)

# # Overall Score: Calculate as (ATS + Content + Structure + Skills + Tone) / 5

# # Be SPECIFIC in all feedback. Instead of "improve your skills section", say "add Python and SQL in skills section as they're required for Data Analyst role" or "move your data analysis skills to the top of your skills list".

# # Keep friendly but professional tone. No markdown formatting.
# # """
    
# #     user_prompt = f"Target Role: {state['target_role']}\n\nResume text: {state['resume_text']}"
    
# #     messages = [
# #         SystemMessage(content=system_prompt),
# #         HumanMessage(content=user_prompt)
# #     ]
    
# #     response = llm.invoke(messages)
# #     return {"messages": [response]}

# # def parse_llm_response(response_text):
# #     """Parse the LLM response and extract JSON"""
# #     try:
# #         # Try to find JSON in the response
# #         json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
# #         if json_match:
# #             json_str = json_match.group(0)
# #             return json.loads(json_str)
# #         else:
# #             return json.loads(response_text)
# #     except json.JSONDecodeError as e:
# #         print(f"JSON Parse Error: {e}")
# #         print(f"Response: {response_text[:500]}")
# #         # Fallback response
# #         return {
# #             "target_role": "Unknown",
# #             "overall_score": 0,
# #             "brief_overview": "Error parsing resume analysis. Please try again.",
# #             "ats_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "missing_keywords": [], "role_relevance": "Unable to determine"},
# #             "content_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "strong_sections": [], "weak_sections": [], "role_alignment": "Unable to determine"},
# #             "structure_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "missing_sections": [], "layout_suggestions": []},
# #             "skills_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "relevant_skills": [], "missing_skills": [], "skills_to_highlight": [], "role_match": "Unable to determine"},
# #             "tone_style_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "language_suggestions": [], "professionalism_notes": "Unable to determine"},
# #             "role_specific_analysis": {"match_percentage": 0, "key_requirements_met": [], "key_requirements_missing": [], "competitive_advantages": [], "red_flags": [], "priority_changes": []},
# #             "career_opportunities": [],
# #             "action_plan": []
# #         }

# # def analyze_resume_with_scores(resume_text, target_role):
# #     graph_builder = StateGraph(State)
# #     graph_builder.add_node("scoring_node", resume_scoring_llm)
# #     graph_builder.add_edge(START, "scoring_node")
# #     graph_builder.add_edge("scoring_node", END)
# #     graph = graph_builder.compile()
    
# #     response = graph.invoke({
# #         "messages": [],
# #         "target_role": target_role,
# #         "resume_text": resume_text
# #     })
    
# #     # Parse the response
# #     llm_output = response["messages"][-1].content
# #     parsed_result = parse_llm_response(llm_output)
    
# #     return parsed_result

# # # FastAPI app
# # app = FastAPI()

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # @app.get("/")
# # async def root():
# #     return {"message": "Resume Analyzer API with Role-Based Scoring is running!"}

# # @app.post("/analyze-resume")
# # async def analyze_resume(
# #     file: UploadFile = File(...),
# #     target_role: str = Form(...)
# # ):
# #     # Validate inputs
# #     if not file.filename.endswith('.pdf'):
# #         return {"success": False, "error": "Only PDF files are supported"}
    
# #     if not target_role or len(target_role.strip()) == 0:
# #         return {"success": False, "error": "Target role is required"}
    
# #     # Save the uploaded file to temp
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
# #         tmp.write(await file.read())
# #         tmp_path = tmp.name

# #     try:
# #         text = extract_text(tmp_path)
        
# #         if not text.strip():
# #             return {"success": False, "error": "No text could be extracted from the PDF"}
        
# #         result = analyze_resume_with_scores(text, target_role.strip())
        
# #         return {
# #             "success": True,
# #             "target_role": result["target_role"],
# #             "overall_score": result["overall_score"],
# #             "brief_overview": result["brief_overview"],
            
# #             "scores": {
# #                 "ats": result["ats_score"],
# #                 "content": result["content_score"],
# #                 "structure": result["structure_score"],
# #                 "skills": result["skills_score"],
# #                 "tone_style": result["tone_style_score"]
# #             },
            
# #             "role_analysis": result["role_specific_analysis"],
# #             "career_opportunities": result["career_opportunities"],
# #             "action_plan": result["action_plan"]
# #         }
    
# #     except Exception as e:
# #         import traceback
# #         return {"success": False, "error": f"Analysis failed: {str(e)}", "traceback": traceback.format_exc()}
    
# #     finally:
# #         if os.path.exists(tmp_path):
# #             os.remove(tmp_path)

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

# load_dotenv()

# # LangGraph State
# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#     target_role: str
#     resume_text: str

# # Request model
# class ResumeAnalysisRequest(BaseModel):
#     target_role: str

# llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)  # Lower temperature for consistency

# def extract_text(pdf_path):
#     doc = fitz.open(pdf_path)
#     all_text = ""
#     for page in doc:
#         all_text += page.get_text()
#     doc.close()
#     return all_text

# def resume_scoring_llm(state: State):
#     system_prompt = """You are an expert career mentor and resume analyzer with 15 years of experience in technical recruitment.

# Your task is to CRITICALLY and OBJECTIVELY analyze the resume for the specific target role. BE HARSH where needed - most resumes have significant room for improvement.

# CRITICAL SCORING RULES:
# 1. ACTUALLY READ AND ANALYZE the resume content - don't give generic scores
# 2. Scores should VARY significantly based on actual content quality
# 3. A score of 60-70 means "acceptable but needs improvement"
# 4. A score of 70-80 means "good with minor issues"
# 5. A score of 80-90 means "excellent, professional quality"
# 6. A score of 90+ should be RARE and only for exceptional resumes
# 7. If something is missing or poor, REFLECT IT IN THE SCORE

# SCORING METHODOLOGY (score 0-100 for each category):

# **ATS Score (Applicant Tracking System Compatibility):**
# - Start at 50 (baseline)
# - SUBTRACT 5-10 points for each missing critical keyword for the target role
# - SUBTRACT 10-15 points for poor formatting (tables, columns, headers/footers)
# - SUBTRACT 5-10 points for missing standard sections (Experience, Education, Skills)
# - ADD 5-10 points for each well-formatted section with proper hierarchy
# - ADD 5-10 points for quantifiable achievements (numbers, percentages, metrics)
# - ADD 5 points for clean, simple formatting

# **Content Score (Relevance & Impact):**
# - Start at 50 (baseline)
# - SUBTRACT 15-20 points if experience is NOT relevant to target role
# - SUBTRACT 10-15 points if bullets are generic (e.g., "responsible for", "worked on")
# - SUBTRACT 10 points for lack of quantifiable results
# - ADD 10-15 points for each highly relevant experience
# - ADD 10 points for strong achievement statements with metrics
# - ADD 5-10 points if experience level matches role requirements

# **Structure Score (Organization & Format):**
# - Start at 50 (baseline)
# - SUBTRACT 10-15 points for poor section ordering (less important stuff first)
# - SUBTRACT 10 points for inconsistent formatting
# - SUBTRACT 5-10 points if resume is too long (>2 pages for <10 years exp)
# - SUBTRACT 5 points for poor readability (dense text, no white space)
# - ADD 10-15 points for clear hierarchy and scannable layout
# - ADD 10 points for consistent formatting throughout
# - ADD 5 points for appropriate length

# **Skills Score (Technical & Soft Skills Relevance):**
# - Start at 50 (baseline)
# - SUBTRACT 20-30 points if CRITICAL skills for the role are MISSING
# - SUBTRACT 10-15 points if skills are listed without demonstration/context
# - SUBTRACT 5-10 points for outdated or irrelevant skills
# - ADD 15-20 points if ALL required skills for role are present
# - ADD 10 points for proper skills demonstration in experience section
# - ADD 5-10 points for good balance of technical and soft skills

# **Tone & Style Score (Professional Communication):**
# - Start at 50 (baseline)
# - SUBTRACT 15-20 points for excessive passive voice or weak verbs
# - SUBTRACT 10 points for grammatical errors or typos
# - SUBTRACT 5-10 points for overly verbose or unclear writing
# - ADD 15-20 points for strong action verbs and clear impact statements
# - ADD 10 points for concise, professional language
# - ADD 5 points for error-free grammar

# **Calculate Overall Score:** (ATS + Content + Structure + Skills + Tone) / 5

# You MUST respond with ONLY a valid JSON object (no markdown, no code blocks, no explanations outside JSON):

# {
#     "target_role": "<the exact role being analyzed>",
#     "overall_score": <calculated average - must be mathematically accurate>,
#     "brief_overview": "<2-3 sentences about resume quality and fit for target role - be honest about gaps>",
    
#     "ats_score": {
#         "score": <0-100, based on actual analysis above>,
#         "feedback": "<2-3 sentences explaining the score>",
#         "issues": ["specific issue found in THIS resume", "another specific issue"],
#         "improvements": ["specific fix for THIS resume", "another specific fix"],
#         "missing_keywords": ["actual keyword from job requirements", "another keyword"],
#         "role_relevance": "<honest assessment of ATS optimization for this specific role>"
#     },
    
#     "content_score": {
#         "score": <0-100, based on actual content quality>,
#         "feedback": "<2-3 sentences about content relevance and impact>",
#         "issues": ["specific content problem in THIS resume", "another problem"],
#         "improvements": ["add metric to bullet point in X section", "rewrite Y achievement to show impact"],
#         "strong_sections": ["actual section that is strong"],
#         "weak_sections": ["actual section that needs work"],
#         "role_alignment": "<specific assessment of how content matches target role>"
#     },
    
#     "structure_score": {
#         "score": <0-100, based on actual structure>,
#         "feedback": "<2-3 sentences about organization>",
#         "issues": ["specific structural problem found", "another problem"],
#         "improvements": ["specific structural fix needed", "another fix"],
#         "missing_sections": ["section needed for this role", "another section"],
#         "layout_suggestions": ["specific layout improvement", "another improvement"]
#     },
    
#     "skills_score": {
#         "score": <0-100, be harsh if skills don't match role>,
#         "feedback": "<2-3 sentences about skills match>",
#         "issues": ["specific skill gap for this role", "another gap"],
#         "improvements": ["add specific skill X with context", "demonstrate skill Y better"],
#         "relevant_skills": ["skill from resume relevant to role", "another skill"],
#         "missing_skills": ["critical missing skill for role", "recommended skill"],
#         "skills_to_highlight": ["skill to emphasize more", "skill to move up"],
#         "role_match": "<honest percentage or description>"
#     },
    
#     "tone_style_score": {
#         "score": <0-100, based on writing quality>,
#         "feedback": "<2-3 sentences about professional tone>",
#         "issues": ["specific language issue found", "another issue"],
#         "improvements": ["replace passive voice in section X", "strengthen verbs in section Y"],
#         "language_suggestions": ["specific rewording for bullet point", "another suggestion"],
#         "professionalism_notes": "<assessment of tone for target role>"
#     },
    
#     "role_specific_analysis": {
#         "match_percentage": <0-100, realistic assessment>,
#         "key_requirements_met": ["requirement actually met in resume", "another met"],
#         "key_requirements_missing": ["requirement not shown in resume", "another missing"],
#         "competitive_advantages": ["actual strength for this role", "another strength"],
#         "red_flags": ["actual concern for recruiters", "another concern"],
#         "priority_changes": ["most critical specific change", "second priority change", "third priority"]
#     },
    
#     "career_opportunities": ["role highly aligned with experience", "alternative role to consider", "growth path role"],
    
#     "action_plan": [
#         {"priority": "High", "action": "specific actionable change", "section": "specific section name"},
#         {"priority": "High", "action": "another critical change", "section": "specific section"},
#         {"priority": "Medium", "action": "important improvement", "section": "specific section"},
#         {"priority": "Low", "action": "nice to have improvement", "section": "specific section"}
#     ]
# }

# REMEMBER: 
# - Scores should be DIFFERENT for different resumes
# - Be SPECIFIC - reference actual content from the resume
# - Be CRITICAL - most resumes are NOT 80+ quality
# - CALCULATE overall_score accurately: (sum of 5 scores) / 5
# """
    
#     user_prompt = f"""Analyze this resume for the role of: {state['target_role']}

# RESUME TEXT:
# {state['resume_text'][:8000]}

# Provide detailed, role-specific analysis with accurate scores based on the resume content."""
    
#     messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=user_prompt)
#     ]
    
#     response = llm.invoke(messages)
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
            
#             # Validate and recalculate overall score for accuracy
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
#         # Fallback response
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

# def analyze_resume_with_scores(resume_text, target_role):
#     graph_builder = StateGraph(State)
#     graph_builder.add_node("scoring_node", resume_scoring_llm)
#     graph_builder.add_edge(START, "scoring_node")
#     graph_builder.add_edge("scoring_node", END)
#     graph = graph_builder.compile()
    
#     response = graph.invoke({
#         "messages": [],
#         "target_role": target_role,
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
#     return {"message": "Resume Analyzer API with Role-Based Scoring is running!"}

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
#         text = extract_text(tmp_path)
        
#         if not text.strip():
#             return {"success": False, "error": "No text could be extracted from the PDF"}
        
#         result = analyze_resume_with_scores(text, target_role.strip())
        
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
from typing import Annotated, Optional
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from fastapi import FastAPI, UploadFile, File, Form
import tempfile
import os
from dotenv import load_dotenv
import uvicorn
import json
import re
from pydantic import BaseModel
import base64

load_dotenv()

# LangGraph State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    target_role: str
    resume_images: list
    resume_text: str  # Keep text as fallback

# Request model
class ResumeAnalysisRequest(BaseModel):
    target_role: str

# Using Llama 4 Maverick with vision capabilities
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3)

def pdf_to_base64_images(pdf_path, dpi=150):
    """Convert PDF pages to base64 encoded images"""
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to image with specified DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img_data = pix.tobytes("png")
        
        # Convert to base64
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        images.append(img_base64)
    
    doc.close()
    return images

def extract_text_fallback(pdf_path):
    """Fallback text extraction"""
    doc = fitz.open(pdf_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text()
    doc.close()
    return all_text

def resume_scoring_llm(state: State):
    system_prompt = """You are an expert career mentor and resume analyzer with 15 years of experience in technical recruitment.

You are analyzing a VISUAL representation of a resume PDF. Pay close attention to:
- Layout and formatting (columns, tables, headers, footers)
- Visual hierarchy and readability
- White space usage and text density
- Font choices and consistency
- Section organization and flow
- ATS-compatibility (avoid complex layouts, graphics in text, tables)
- Professional appearance

Your task is to CRITICALLY and OBJECTIVELY analyze this resume for the specific target role. BE HARSH where needed - most resumes have significant room for improvement.

CRITICAL SCORING RULES:
1. ACTUALLY ANALYZE the resume - look at both visual formatting AND content
2. Scores should VARY significantly based on actual quality
3. Score 60-70 = "acceptable but needs improvement"
4. Score 70-80 = "good with minor issues"
5. Score 80-90 = "excellent, professional quality"
6. Score 90+ = RARE, only for exceptional resumes
7. If something is missing or poor, REFLECT IT IN THE SCORE

SCORING METHODOLOGY (0-100 for each category):

**ATS Score (Applicant Tracking System Compatibility):**
Start at 50 (baseline)
- SUBTRACT 15-20 for complex layouts (multi-column, text boxes, headers/footers with info)
- SUBTRACT 10-15 for each missing critical keyword for target role
- SUBTRACT 10 for graphics, images, or charts
- SUBTRACT 5-10 for missing standard sections
- ADD 10-15 for clean, single-column layout
- ADD 10 for quantifiable achievements with metrics
- ADD 5-10 for proper section hierarchy

**Content Score (Relevance & Impact):**
Start at 50 (baseline)
- SUBTRACT 20-25 if experience is NOT relevant to target role
- SUBTRACT 15-20 for generic bullets ("responsible for", "worked on")
- SUBTRACT 10-15 for lack of quantifiable results
- ADD 15-20 for highly relevant experience matching role requirements
- ADD 15 for strong achievement statements with specific metrics
- ADD 10 if experience level matches role needs

**Structure Score (Organization & Visual Format):**
Start at 50 (baseline)
- SUBTRACT 15-20 for poor visual hierarchy or cluttered layout
- SUBTRACT 10-15 for inconsistent formatting (fonts, spacing, alignment)
- SUBTRACT 10 if resume is too long (>2 pages for <10 years exp) or too short
- SUBTRACT 10 for poor readability (dense text, no white space)
- ADD 15-20 for excellent visual hierarchy and scannable layout
- ADD 10-15 for consistent, professional formatting
- ADD 5-10 for appropriate length and good use of white space

**Skills Score (Technical & Soft Skills for Role):**
Start at 50 (baseline)
- SUBTRACT 25-35 if CRITICAL required skills for role are MISSING
- SUBTRACT 15-20 if skills listed without demonstration in experience
- SUBTRACT 10 for irrelevant or outdated skills prominently featured
- ADD 20-25 if ALL required skills for role are present
- ADD 15 for clear skills demonstration with context/examples
- ADD 10 for excellent balance of technical and soft skills

**Tone & Style Score (Professional Communication):**
Start at 50 (baseline)
- SUBTRACT 20-25 for excessive passive voice or weak verbs
- SUBTRACT 15 for grammatical errors, typos, or inconsistencies
- SUBTRACT 10 for overly verbose or unclear writing
- ADD 20-25 for strong action verbs and clear impact statements
- ADD 15 for concise, professional, error-free language
- ADD 10 for appropriate tone for target role level

**Overall Score = (ATS + Content + Structure + Skills + Tone) / 5**

RESPOND WITH ONLY VALID JSON (no markdown, no code blocks):

{
    "target_role": "<exact target role>",
    "overall_score": <calculated average>,
    "brief_overview": "<2-3 honest sentences about resume quality and role fit>",
    
    "ats_score": {
        "score": <0-100>,
        "feedback": "<2-3 sentences explaining score with specific visual observations>",
        "issues": ["specific formatting issue seen in resume", "another specific issue"],
        "improvements": ["specific fix based on what you see", "another fix"],
        "missing_keywords": ["actual keyword needed for role", "another keyword"],
        "role_relevance": "<honest ATS assessment for this role>"
    },
    
    "content_score": {
        "score": <0-100>,
        "feedback": "<2-3 sentences about content quality and relevance>",
        "issues": ["specific content problem observed", "another problem"],
        "improvements": ["add specific metric to section X", "rewrite bullet Y to show impact"],
        "strong_sections": ["actual strong section name"],
        "weak_sections": ["actual weak section name"],
        "role_alignment": "<specific assessment for target role>"
    },
    
    "structure_score": {
        "score": <0-100>,
        "feedback": "<2-3 sentences about visual structure and organization>",
        "issues": ["specific visual/structural problem", "another problem"],
        "improvements": ["specific layout improvement", "another improvement"],
        "missing_sections": ["section needed for role", "another section"],
        "layout_suggestions": ["specific visual improvement", "another suggestion"]
    },
    
    "skills_score": {
        "score": <0-100>,
        "feedback": "<2-3 sentences about skills match to role>",
        "issues": ["specific skill gap for this role", "another gap"],
        "improvements": ["add specific skill X with context", "demonstrate skill Y"],
        "relevant_skills": ["skill from resume relevant to role"],
        "missing_skills": ["critical missing skill", "recommended skill"],
        "skills_to_highlight": ["skill to emphasize", "skill to move up"],
        "role_match": "<honest percentage or assessment>"
    },
    
    "tone_style_score": {
        "score": <0-100>,
        "feedback": "<2-3 sentences about writing quality>",
        "issues": ["specific language issue", "another issue"],
        "improvements": ["specific rewording suggestion", "another suggestion"],
        "language_suggestions": ["replace passive voice in X", "strengthen verbs in Y"],
        "professionalism_notes": "<tone assessment for role>"
    },
    
    "role_specific_analysis": {
        "match_percentage": <0-100>,
        "key_requirements_met": ["requirement actually shown", "another met"],
        "key_requirements_missing": ["requirement not demonstrated", "another missing"],
        "competitive_advantages": ["actual strength for role", "another strength"],
        "red_flags": ["actual concern for recruiters", "another concern"],
        "priority_changes": ["most critical change", "second priority", "third priority"]
    },
    
    "career_opportunities": ["highly aligned role", "alternative role", "growth path role"],
    
    "action_plan": [
        {"priority": "High", "action": "specific actionable change", "section": "section name"},
        {"priority": "High", "action": "another critical change", "section": "section name"},
        {"priority": "Medium", "action": "important improvement", "section": "section name"},
        {"priority": "Low", "action": "enhancement", "section": "section name"}
    ]
}

BE SPECIFIC - reference what you actually see in the resume visually and content-wise.
BE CRITICAL - most resumes are NOT 80+ quality.
VARY SCORES - different resumes should get different scores based on quality."""
    
    # Build message content with images
    content_parts = [
        {
            "type": "text",
            "text": f"""Analyze this resume for the role of: {state['target_role']}

Look carefully at the visual formatting, layout, and content. Be specific about what you observe."""
        }
    ]
    
    # Add each page image
    for idx, img_base64 in enumerate(state['resume_images']):
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}"
            }
        })
    
    # Add text as additional context if needed
    if state.get('resume_text'):
        content_parts.append({
            "type": "text",
            "text": f"\n\nResume Text Content (for reference):\n{state['resume_text'][:6000]}"
        })
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content_parts)
    ]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

def parse_llm_response(response_text):
    """Parse the LLM response and extract JSON"""
    try:
        # Remove markdown code blocks if present
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\n', '', cleaned)
            cleaned = re.sub(r'\n```$', '', cleaned)
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            
            # Validate and recalculate overall score
            scores = [
                parsed.get("ats_score", {}).get("score", 0),
                parsed.get("content_score", {}).get("score", 0),
                parsed.get("structure_score", {}).get("score", 0),
                parsed.get("skills_score", {}).get("score", 0),
                parsed.get("tone_style_score", {}).get("score", 0)
            ]
            
            if all(isinstance(s, (int, float)) for s in scores):
                calculated_overall = round(sum(scores) / 5, 1)
                parsed["overall_score"] = calculated_overall
            
            return parsed
        else:
            return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Response: {response_text[:500]}")
        return {
            "target_role": "Unknown",
            "overall_score": 0,
            "brief_overview": "Error parsing resume analysis. Please try again.",
            "ats_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "missing_keywords": [], "role_relevance": "Unable to determine"},
            "content_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "strong_sections": [], "weak_sections": [], "role_alignment": "Unable to determine"},
            "structure_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "missing_sections": [], "layout_suggestions": []},
            "skills_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "relevant_skills": [], "missing_skills": [], "skills_to_highlight": [], "role_match": "Unable to determine"},
            "tone_style_score": {"score": 0, "feedback": "Unable to analyze", "issues": [], "improvements": [], "language_suggestions": [], "professionalism_notes": "Unable to determine"},
            "role_specific_analysis": {"match_percentage": 0, "key_requirements_met": [], "key_requirements_missing": [], "competitive_advantages": [], "red_flags": [], "priority_changes": []},
            "career_opportunities": [],
            "action_plan": []
        }

def analyze_resume_with_scores(resume_images, resume_text, target_role):
    graph_builder = StateGraph(State)
    graph_builder.add_node("scoring_node", resume_scoring_llm)
    graph_builder.add_edge(START, "scoring_node")
    graph_builder.add_edge("scoring_node", END)
    graph = graph_builder.compile()
    
    response = graph.invoke({
        "messages": [],
        "target_role": target_role,
        "resume_images": resume_images,
        "resume_text": resume_text
    })
    
    # Parse the response
    llm_output = response["messages"][-1].content
    parsed_result = parse_llm_response(llm_output)
    
    return parsed_result

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Resume Analyzer API with Vision-Based Scoring is running!"}

@app.post("/analyze-resume")
async def analyze_resume(
    file: UploadFile = File(...),
    target_role: str = Form(...)
):
    # Validate inputs
    if not file.filename.endswith('.pdf'):
        return {"success": False, "error": "Only PDF files are supported"}
    
    if not target_role or len(target_role.strip()) == 0:
        return {"success": False, "error": "Target role is required"}
    
    # Save the uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Convert PDF to images for vision analysis
        resume_images = pdf_to_base64_images(tmp_path, dpi=150)
        
        # Also extract text as fallback/additional context
        resume_text = extract_text_fallback(tmp_path)
        
        if not resume_images:
            return {"success": False, "error": "Could not process PDF pages"}
        
        if not resume_text.strip():
            return {"success": False, "error": "No text could be extracted from the PDF"}
        
        result = analyze_resume_with_scores(resume_images, resume_text, target_role.strip())
        
        return {
            "success": True,
            "target_role": result["target_role"],
            "overall_score": result["overall_score"],
            "brief_overview": result["brief_overview"],
            
            "scores": {
                "ats": result["ats_score"],
                "content": result["content_score"],
                "structure": result["structure_score"],
                "skills": result["skills_score"],
                "tone_style": result["tone_style_score"]
            },
            
            "role_analysis": result["role_specific_analysis"],
            "career_opportunities": result["career_opportunities"],
            "action_plan": result["action_plan"]
        }
    
    except Exception as e:
        import traceback
        return {"success": False, "error": f"Analysis failed: {str(e)}", "traceback": traceback.format_exc()}
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)