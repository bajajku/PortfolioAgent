from langchain_core.tools import tool
from utils.retriever import global_retriever

@tool
def get_project_details(project_name: str) -> str:
    """
    Get detailed information about a specific project from Kunal's portfolio.
    
    Args:
        project_name: The name of the project to look up (e.g., "Vercel Clone", "Groovify")
        
    Returns:
        str: Detailed information about the specified project.
    """
    # Search with project name as query
    docs = global_retriever.get_relevant_documents(project_name)
    
    if not docs:
        return f"I couldn't find information about a project called '{project_name}' in Kunal's portfolio."
    
    results = "\n\n".join([doc.page_content for doc in docs])
    return f"Here are the details about {project_name}:\n\n{results}"