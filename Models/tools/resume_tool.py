from typing import Dict, List, Optional
from langchain_core.tools import tool
from utils.pdf_loader import ResumeProcessor
from utils.text_loader import load_and_process_text

@tool
def search_resume( query: str) -> str:
    """
    Search Kunal's resume for information related to the query.
    
    Args:
        query: The question or search term about Kunal's professional background.
        
    Returns:
        str: Relevant information from Kunal's resume.
    """
    # Get relevant documents
    docs = load_and_process_text().get_relevant_documents(query)
    
    if not docs:
        return "I couldn't find specific information about that in Kunal's resume."
    
    # Combine the content from relevant documents
    results = "\n\n".join([doc.page_content for doc in docs])
    print(f"Results: {results}")
    return f"Found the following information in Kunal's resume:\n\n{results}"