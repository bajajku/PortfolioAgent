from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from Models.agent import PortfolioAgent
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Check environment variables
HF_API_KEY = os.environ.get("HF_API_KEY")
REPO_ID = os.environ.get("REPO_ID")
RESUME_PATH = os.environ.get("RESUME_PATH", "resume.pdf")

if not HF_API_KEY:
    raise EnvironmentError("Missing environment variable: HF_API_KEY")
if not REPO_ID:
    raise EnvironmentError("Missing environment variable: REPO_ID")

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id=REPO_ID,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HF_API_KEY
)

chat = ChatHuggingFace(llm=llm, verbose=True)

# Initialize the agent
agent = PortfolioAgent(chat, RESUME_PATH)

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