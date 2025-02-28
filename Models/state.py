from typing import Annotated, List, Dict, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    """State for the portfolio assistant agent."""
    messages: Annotated[list[AnyMessage], add_messages]
    next_steps: Optional[List[str]]
    current_tool: Optional[str]
    tools_used: List[str]