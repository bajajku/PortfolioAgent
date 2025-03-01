import getpass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
from Models.assistant import Assistant
import dotenv
from langchain_core.messages import ToolMessage, HumanMessage
from Models.prompt import AGENT_PROMPT
from Models.state import State
from Models.tools.helper import create_tool_node_with_fallback
from Models.tools.resume_tool import search_resume
from Models.tools.project_tool import get_project_details
from Models.tools.contact_tool import get_contact_info
from Models.tools.skills_tool import assess_skills_for_role
from langgraph.graph import StateGraph, START, END
from utils.retriever import global_retriever

# Load environment variables
dotenv.load_dotenv()

if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

# Ensure retriever is initialized once at application startup
from langchain.chat_models import init_chat_model
tools = [get_contact_info, search_resume, get_project_details, assess_skills_for_role]
model = init_chat_model(os.environ.get("MISTRAL_MODEL"), model_provider="mistralai")
tools_names = {
"get_contact_info": get_contact_info,
"search_resume": search_resume,
"get_project_details": get_project_details,
"assess_skills_for_role": assess_skills_for_role
}

def execute_tools(state: State):
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:

        if not t['name'] in tools_names:
            result = "Error: There's no such tool, please try again"
        else:
            result = tools_names[t['name']].invoke(t['args'])

            results.append(
            ToolMessage(
                tool_call_id=t['id'],
                name=t['name'],
                content=str(result)
            ))

    return {'messages': results}

def tool_exists(state: State):
    result = state['messages'][-1]
    return len(result.tool_calls) > 0

prompt = AGENT_PROMPT
RUNNABLE = prompt | model.bind_tools(tools)

builder = StateGraph(State)
builder.add_node("assistant", Assistant(RUNNABLE))
builder.add_node("tools", create_tool_node_with_fallback(tools))
builder.add_conditional_edges(
"assistant",
    tool_exists,
{True: "tools", False: END}
)
builder.add_edge("tools", "assistant")
builder.set_entry_point("assistant")

graph = builder.compile()
    

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
        # Use the graph instead of agent
        messages = [HumanMessage(content=query.question)]
        result = graph.invoke({"messages": messages})

        print(result)
        
        # Extract the assistant's response
        assistant_responses = [msg.content for msg in result["messages"] if msg.type == "ai"]
        
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