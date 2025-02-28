from langchain_core.tools import tool
from utils.text_loader import load_and_process_text


@tool
def assess_skills_for_role(role_description: str) -> str:
    """
    Assess Kunal's skills and experience in relation to a specific role or job description.
    
    Args:
        role_description: Description of the role or skills to assess
        
    Returns:
        str: Assessment of how Kunal's skills match the requirements.
    """
    # Get skills section
    docs = load_and_process_text().get_relevant_documents("technical skills experience")
    
    if not docs:
        return "I couldn't find detailed skills information in Kunal's resume."
    
    skills_info = "\n\n".join([doc.page_content for doc in docs])
    
    # Now look for matches with the requested role
    role_matches = load_and_process_text().get_relevant_documents(role_description)
    
    role_matches_text = "\n\n".join([doc.page_content for doc in role_matches]) if role_matches else ""
    
    return f"""
    Skills Assessment for {role_description}:
    
    Kunal's Technical Skills:
    {skills_info}
    
    Relevant Experience:
    {role_matches_text}
    
    Based on Kunal's resume, he has experience and skills that align with this role.
    """