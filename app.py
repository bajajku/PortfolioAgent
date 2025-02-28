from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
from Models.agent import PortfolioAgent
from Models.llm import LLM
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Check environment variables
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek/deepseek-r1")
RESUME_PATH = os.environ.get("RESUME_PATH", "resume.pdf")

if not OPENROUTER_API_KEY:
    raise EnvironmentError("Missing environment variable: OPENROUTER_API_KEY")

# Initialize the LLM
llm_wrapper = LLM(
    model_name=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    temperature=0.7,
    streaming=True,
    base_url="https://openrouter.ai/api/v1"
)

# Initialize the agent
agent = PortfolioAgent(llm_wrapper, RESUME_PATH)

# Initialize FastAPI app
app = FastAPI(title="Kunal's Portfolio Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(query: QueryRequest):
    try:
        messages = agent.invoke(query.question)
        
        # Extract the assistant's response
        assistant_responses = [msg.content for msg in messages if msg.type == "ai"]
        
        if not assistant_responses:
            return ChatResponse(answer="No response generated.")
            
        return ChatResponse(answer=assistant_responses[-1])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)