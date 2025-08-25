from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import uuid
import shutil
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Lazily import the pipeline inside the endpoint to avoid startup errors when heavy
# dependencies (like sentence-transformers) are not installed in the environment.

app = FastAPI()

# Add CORS so frontends on Render and local hosts can call the API.
# To allow all origins on Render while still permitting credentials,
# use allow_origin_regex. This is intentionally wide-open; if you
# want stricter security, replace the regex with a specific list.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Concurrency control: limit number of concurrent heavy tasks (blocking LLM or embedding pipelines)
# Default to 6 workers, configurable via WORKER_COUNT env var
WORKER_COUNT = int(os.getenv("WORKER_COUNT", "6"))
semaphore = asyncio.BoundedSemaphore(WORKER_COUNT)

# Load environment variables early so ADC and API keys are present when importing pipeline
load_dotenv()

# --- Pydantic Models (Unchanged, but shown for context) ---
class Question(BaseModel):
    id: str
    question: str
    answer: Optional[str] = ""

class Section(BaseModel):
    title: str
    questions: List[Question]

class Original(BaseModel):
    sections: List[Section]

class StudentQuestion(BaseModel):
    questionId: str
    questionText: str
    referenceAnswer: str
    studentAnswer: Optional[str] = None

class EvaluationRequest(BaseModel):
    original: Original
    questions: List[StudentQuestion]

class QuestionEvaluation(BaseModel):
    id: str
    question: str
    referenceAnswer: Optional[str] = ""
    score: int = Field(..., description="Score for the question, from 0 to 100")
    confidence: Optional[float] = Field(None, description="Model's confidence in the score, from 0 to 1")
    feedback: str = Field(..., description="Specific, constructive feedback for the student's answer")
    suggestions: List[str] = Field(..., description="Actionable suggestions for improvement")

class SectionEvaluation(BaseModel):
    title: str
    score: int = Field(..., description="Average score for this section, from 0 to 100")
    improvements: str = Field(..., description="General areas of improvement for this section")
    suggestions: List[str] = Field(..., description="High-level suggestions for the section")
    questions: List[QuestionEvaluation]

class EvaluationResponse(BaseModel):
    overallScore: int = Field(..., description="The final, overall score for the entire test, from 0 to 100")
    evaluatedAt: str  # ISO8601 string, will be added by our code
    sections: List[SectionEvaluation]

# NEW: A Pydantic model representing only what the LLM should generate
class LLMEvaluationResult(BaseModel):
    overallScore: int = Field(..., description="The final, overall score for the entire test, from 0 to 100")
    sections: List[SectionEvaluation]

# --- Endpoints ---

@app.post("/generate")
async def generate_questions(resume: UploadFile = File(...)):
    if resume.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Use a unique filename to avoid collisions when multiple users upload files simultaneously
    unique_name = f"{uuid.uuid4().hex}_{resume.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(resume.file, buffer)
    
    try:
        try:
            from endeavor_rag_service import interview_rag_pipeline, collection
        except ModuleNotFoundError as e:
            raise HTTPException(status_code=500, detail=f"Missing dependency: {e}. Please run: pip install -r requirements.txt")

        # Limit concurrency to a bounded number of workers and run the blocking pipeline in a thread
        await semaphore.acquire()
        try:
            result = await asyncio.to_thread(interview_rag_pipeline, file_path, collection)
        finally:
            semaphore.release()

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
@app.get("/health")
async def health():
    """Health check used by frontends.

    Returns 200 when the service appears healthy (Google credentials present),
    or 401 when credentials are missing. The frontend can treat the response
    as healthy when response.status === 401 || response.ok (200-299).
    """
    # Acceptable credentials: either an API key or a service account file path
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")):
        return JSONResponse(status_code=401, content={"ok": False, "detail": "Missing Google API credentials"})
    return JSONResponse(status_code=200, content={"ok": True})


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_answers(content: EvaluationRequest):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.exceptions import OutputParserException

        # 1. Use a dedicated Output Parser for robust JSON handling
        parser = JsonOutputParser(pydantic_object=LLMEvaluationResult)

        prompt_template = """You are a meticulous and fair interview evaluator. Your task is to assess a candidate's responses based on a provided set of questions and reference answers.

        **Reference Material (Questions and Ideal Answers):**
        {original_content}

        **Candidate's Submissions:**
        {student_answers}

        Please evaluate the candidate's answers critically. For each question, provide a score from 0-100, specific feedback, and actionable suggestions. Then, calculate section scores and an overall score.

        Return your complete evaluation strictly in the required JSON format. Do not add any extra text, explanations, or markdown formatting around the JSON object.

        {format_instructions}
        """

        # Format the original content
        original_content = ""
        for section in content.original.sections:
            original_content += f"\nSection: {section.title}\n"
            for q in section.questions:
                original_content += f"  Q{q.id}: {q.question}\n"
                if q.answer:
                    original_content += f"  Reference Answer: {q.answer}\n"

        # Format student answers
        student_answers = ""
        for q in content.questions:
            student_answers += f"\nQuestion ID: {q.questionId}\n"
            student_answers += f"  - Question: {q.questionText}\n"
            student_answers += f"  - Student Answer: {q.studentAnswer or 'No answer provided'}\n"

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["original_content", "student_answers"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Make sure GOOGLE_API_KEY is set in your .env file
        model = ChatGoogleGenerativeAI(model='gemma-3-27b-it', temperature=0.2)
        
        # 2. Chain the prompt, model, and parser together
        chain = prompt | model | parser
        
        try:
            # 3. Invoke the chain in a thread while holding the semaphore to limit concurrency.
            await semaphore.acquire()
            try:
                llm_result = await asyncio.to_thread(
                    chain.invoke,
                    {
                        "original_content": original_content,
                        "student_answers": student_answers,
                    },
                )
            finally:
                semaphore.release()

            # 4. Construct the final response object, adding the timestamp
            final_response = EvaluationResponse(
                **llm_result,
                evaluatedAt=datetime.utcnow().isoformat() + "Z"
            )
            # 5. Return the Pydantic object directly. FastAPI handles validation and serialization.
            return final_response

        except OutputParserException as e:
            # This error is much more specific than a generic json.JSONDecodeError
            print(f"Error parsing LLM output: {e}") # Crucial for debugging
            raise HTTPException(
                status_code=500,
                detail="The AI evaluator failed to return a valid structured response. Please try again."
            )
            
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {e}. Please install langchain-google-genai.")
    except Exception as e:
        # Catches other errors, like authentication failure with Google AI
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during evaluation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)