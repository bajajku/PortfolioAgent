from typing import Dict, List, Tuple, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from Models.state import State
from Models.prompt import AGENT_PROMPT
from Models.tools.resume_tool import ResumeSearchTool
from Models.tools.project_tool import ProjectInfoTool
from Models.tools.contact_tool import ContactInfoTool
from Models.tools.skills_tool import SkillsAssessmentTool
from utils.pdf_loader import ResumeProcessor

class PortfolioAgent:
    def __init__(self, llm, resume_path: str):
        """
        Initialize the portfolio agent with tools.
        
        Args:
            llm: Language model to use
            resume_path: Path to the resume PDF
        """
        self.llm = llm
        self.resume_processor = ResumeProcessor(resume_path)
        self.resume_processor.load_and_process()
        
        # Initialize tools
        self.resume_tool = ResumeSearchTool(self.resume_processor)
        self.project_tool = ProjectInfoTool(self.resume_processor)
        self.contact_tool = ContactInfoTool()
        self.skills_tool = SkillsAssessmentTool(self.resume_processor)
        
        # Define tools
        self.tools = [
            self.resume_tool.search_resume,
            self.project_tool.get_project_details,
            self.contact_tool.get_contact_info,
            self.skills_tool.assess_skills_for_role
        ]
        
        # Create the agent using LangGraph
        self.agent = self._create_agent()
        
    def _create_agent(self) -> StateGraph:
        """Create the agent graph using LangGraph"""
        # Create the runnable agent with tools
        runnable = AGENT_PROMPT | self.llm.bind_tools(self.tools)
        
        # Define the graph
        builder = StateGraph(State)
        
        # Add nodes
        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", ToolNode(self.tools))
        
        # Add edges
        builder.add_edge("agent", "tools")
        builder.add_edge("tools", "agent")
        
        # Add conditional edges
        builder.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # Compile the graph
        return builder.compile()
    
    def _agent_node(self, state: State) -> Dict:
        """Agent node that processes messages and decides next steps"""
        # Create a runnable with the LLM and tools
        runnable = AGENT_PROMPT | self.llm.bind_tools(self.tools)
        
        # Run the LLM
        result = runnable.invoke(state)
        
        # Return the updated state
        return {"messages": [result]}
    
    def _should_continue(self, state: State) -> str:
        """Determine if we should continue using tools or end the conversation"""
        # Get the last message
        last_message = state["messages"][-1]
        
        # If there are tool calls, continue
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"
    
    def invoke(self, message: str) -> List[BaseMessage]:
        """
        Process a user message and return the conversation.
        
        Args:
            message: User's message
            
        Returns:
            List of messages in the conversation
        """
        # Initialize the state with the user message
        state = {"messages": [HumanMessage(content=message)], "tools_used": []}
        
        # Run the agent
        result = self.agent.invoke(state)
        
        # Return the messages
        return result["messages"]