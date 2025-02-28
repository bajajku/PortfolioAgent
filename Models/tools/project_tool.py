from langchain_core.tools import tool
from utils.text_loader import load_and_process_text

class ProjectInfoTool:
    def __init__(self):
        self.retriever = load_and_process_text()

        
    @tool
    def get_project_details(self, project_name: str) -> str:
        """
        Get detailed information about a specific project from Kunal's portfolio.
        
        Args:
            project_name: The name of the project to look up (e.g., "Vercel Clone", "Groovify")
            
        Returns:
            str: Detailed information about the specified project.
        """
        # Search with project name as query
        docs = self.retriever.get_relevant_documents(project_name)
        
        if not docs:
            return f"I couldn't find information about a project called '{project_name}' in Kunal's portfolio."
        
        results = "\n\n".join([doc.page_content for doc in docs])
        return f"Here are the details about {project_name}:\n\n{results}"