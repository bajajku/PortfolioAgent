import os
import getpass
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from Models.agent import PortfolioAgent
import dotenv

def main():
    dotenv.load_dotenv()
    
    # Check environment variables
    HF_API_KEY = os.environ.get("HF_API_KEY")
    REPO_ID = os.environ.get("REPO_ID")
    RESUME_PATH = os.environ.get("RESUME_PATH", "resume.pdf")
    
    if not HF_API_KEY:
        raise EnvironmentError("Missing environment variable: HF_API_KEY")
    if not REPO_ID:
        raise EnvironmentError("Missing environment variable: REPO_ID")
    
    try:
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
        
        # Start conversation loop
        print("Welcome to Kunal's Portfolio Assistant! Ask anything about Kunal's experience, projects, or skills.")
        print("Type 'exit' to end the conversation.")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Thank you for exploring Kunal's portfolio. Goodbye!")
                break
                
            # Process the user input
            messages = agent.invoke(user_input)
            
            # Print the assistant's response
            for message in messages:
                if message.type == "ai":
                    print(f"\nAssistant: {message.content}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()