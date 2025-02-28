from langchain_core.tools import tool


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
