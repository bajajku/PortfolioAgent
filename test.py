import os
import dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from Models.agent import PortfolioAgent
from Models.assistant import Assistant
from Models.llm import LLM
import getpass
from Models.prompt import AGENT_PROMPT
from Models.state import State
from Models.tools.helper import create_tool_node_with_fallback
from Models.tools.resume_tool import ResumeSearchTool
from Models.tools.project_tool import ProjectInfoTool
from Models.tools.contact_tool import ContactInfoTool
from Models.tools.skills_tool import SkillsAssessmentTool
from langgraph.graph import StateGraph, START, END
from utils.pdf_loader import ResumeProcessor
from langgraph.prebuilt import tools_condition
from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

def test_basic_query():
    dotenv.load_dotenv()
    
    @tool
    def get_contact_info() -> str:
        """
        Get Kunal's contact information.
        
        Returns:
            str: Kunal's contact details including email, phone, LinkedIn, and GitHub.
        """
        return """
        Kunal Bajaj's Contact Information:
        - Phone: 437-269-7678
        - Email: kunalbajaj20220@gmail.com
        - LinkedIn: https://ca.linkedin.com/in/kunal-bajaj1
        - GitHub: https://github.com/bajajku
        """


    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

    from langchain.chat_models import init_chat_model
    tools = [get_contact_info]
    model = init_chat_model("mistral-large-2411", model_provider="mistralai")
    tools_names = {
    "get_contact_info": get_contact_info
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
                )
                )

        return {'messages': results}

    def tool_exists(state: State):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0
    def run_llm(state: State):
        messages = state['messages']
        message = model.invoke(messages)
        return {'messages': [message]}

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

    # Initialize the agent
    # agent = PortfolioAgent(model, RESUME_PATH)
    
    # Test queries that should trigger different tools
    # test_queries = [
    #     "What is Kunal's education background?",
    #     "Tell me about the Vercel Clone project",
    #     "How can I contact Kunal?",
    #     "Would Kunal be a good fit for a Machine Learning Engineer role?"
    # ]
    # for query in test_queries:
    #     print(f"\n\n--- Testing query: {query} ---")
    #     messages = graph.invoke({"messages": [{"role": "user", "content": query}]})
    #     print(f"Messages: {messages}")
        
    messages = [HumanMessage(content="What is Kunal's contact information?")]
    result = graph.invoke({"messages": messages})
    print(result)

if __name__ == "__main__":
    test_basic_query()