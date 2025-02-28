from langchain_openai import ChatOpenAI


class LLM:

    def __init__(self, model_name: str, api_key: str, **kwargs):
        """
        Initialize the OpenRouter API client with custom parameters.
        
        Args:
            model_name (str): The model identifier (e.g., "deepseek/deepseek-r1")
            api_key (str): The OpenRouter API token
            **kwargs: Additional parameters for configuration
        """
        self.model_name = model_name
        self.api_key = api_key
        self.config = {
            'temperature': kwargs.get('temperature', 0.7),
            'model': model_name,
            'streaming': kwargs.get('streaming', True),
            'base_url': kwargs.get('base_url', "https://openrouter.ai/api/v1")
        }
        self.client = self.create_llm()


    def create_llm(self):
        """Set up the OpenRouter API endpoint."""
        return ChatOpenAI(
            **self.config,
            api_key=self.api_key
        )
    
    def create_chat(self):
        return self.client