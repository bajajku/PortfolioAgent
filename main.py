import os
import getpass
from Models.agent import PortfolioAgent
from Models.llm import LLM
import dotenv

def main():
    dotenv.load_dotenv()
    
    # Check environment variables
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek/deepseek-r1")
    RESUME_PATH = os.environ.get("RESUME_PATH", "resume.pdf")
    
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("Missing environment variable: OPENROUTER_API_KEY")
    
    try:
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