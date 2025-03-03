# from typing import Dict, List, Tuple, Any, Optional
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# from langchain_core.runnables import Runnable
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from Models.state import State
# from Models.prompt import AGENT_PROMPT
# from Models.tools.resume_tool import search_resume
# from Models.tools.project_tool import get_project_details
# from Models.tools.contact_tool import get_contact_info
# from Models.tools.skills_tool import assess_skills_for_role
# from utils.pdf_loader import ResumeProcessor
# from langgraph.graph import START

# class PortfolioAgent:
#     def __init__(self, llm, resume_path: str):
#         """
#         Initialize the portfolio agent with tools.
        
#         Args:
#             llm: Language model to use
#             resume_path: Path to the resume PDF
#         """
#         self.llm = llm
#         self.resume_processor = ResumeProcessor(resume_path)
#         self.resume_processor.load_and_process()
        
#         # Initialize tools
#         self.resume_tool = search_resume
#         self.project_tool = get_project_details
#         self.contact_tool = get_contact_info
#         self.skills_tool = assess_skills_for_role
        
#         # Define tools
#         self.tools = [
#             self.resume_tool.search_resume,
#             self.project_tool.get_project_details,
#             self.contact_tool.get_contact_info,
#             self.skills_tool.assess_skills_for_role
#         ]
        
#         # Create the agent using LangGraph
#         self.agent = self._create_agent()
        
#     def _create_agent(self) -> StateGraph:
#         """Create the agent graph using LangGraph"""
#         # Create the runnable agent with tools
#         runnable = AGENT_PROMPT | self.llm.bind_tools(self.tools)
        
#         # Define the graph
#         builder = StateGraph(State)
        
#         # Add nodes
#         builder.add_node("agent", self._agent_node)
#         builder.add_node("tools", ToolNode(self.tools))
        
#         # Add edge from START to agent
#         builder.add_edge(START, "agent")  # Add this line to define the entry point
        
#         # Add edges
#         builder.add_edge("agent", "tools")
#         builder.add_edge("tools", "agent")
        
#         # Add conditional edges
#         builder.add_conditional_edges(
#             "agent",
#             self._should_continue,
#             {
#                 "continue": "tools",
#                 "end": END
#             }
#         )
        
#         # Compile the graph
#         return builder.compile()
    
#     def _agent_node(self, state: State) -> Dict:
#         """Agent node that processes messages and decides next steps"""
#         # Create a runnable with the LLM and tools
#         runnable = AGENT_PROMPT | self.llm.bind_tools(self.tools)
        
#         # Run the LLM
#         result = runnable.invoke(state)
        
#         # Return the updated state
#         return {"messages": [result]}
    
#     def _should_continue(self, state: State) -> str:
#         """Determine if we should continue using tools or end the conversation"""
#         # Get the last message
#         last_message = state["messages"][-1]
        
#         # If there are tool calls, continue
#         if hasattr(last_message, "tool_calls") and last_message.tool_calls:
#             return "continue"
#         return "end"
    
#     def invoke(self, message: str) -> List[BaseMessage]:
#         """
#         Process a user message and return the conversation.
        
#         Args:
#             message: User's message
            
#         Returns:
#             List of messages in the conversation
#         """
#         # Initialize the state with the user message
#         state = {"messages": [HumanMessage(content=message)], "tools_used": []}
        
#         # Run the agent
#         result = self.agent.invoke(state)
        
#         # Return the messages
#         return result["messages"]