import os
import dotenv
from Models.agent import PortfolioAgent
from Models.llm import LLM

def test_basic_query():
    dotenv.load_dotenv()
    
    # Set up the model
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek/deepseek-r1")
    RESUME_PATH = os.environ.get("RESUME_PATH", "resume.pdf")
    
    # Initialize the LLM
    llm_wrapper = LLM(
        model_name=MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
        temperature=0.7,
        streaming=True,
        base_url="https://openrouter.ai/api/v1"
    )
    
    chat = llm_wrapper.create_chat()
    
    # Initialize the agent
    agent = PortfolioAgent(chat, RESUME_PATH)
    
    # Test queries that should trigger different tools
    test_queries = [
        "What is Kunal's education background?",
        "Tell me about the Vercel Clone project",
        "How can I contact Kunal?",
        "Would Kunal be a good fit for a Machine Learning Engineer role?"
    ]
    
    for query in test_queries:
        print(f"\n\n--- Testing query: {query} ---")
        messages = agent.invoke(query)
        
        # Print responses
        for i, message in enumerate(messages):
            print(f"\nMessage {i+1}: {message.type}")
            print(f"Content: {message.content}")
            
            # Print tool calls if any
            if hasattr(message, "tool_calls") and message.tool_calls:
                print("\nTool Calls:")
                for tc in message.tool_calls:
                    print(f"  Tool: {tc.name}")
                    print(f"  Input: {tc.args}")

if __name__ == "__main__":
    test_basic_query()