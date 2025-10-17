"""LLM model wrappers using LangChain"""

from langchain_openai import ChatOpenAI


class LLMs(object):
    """Wrapper for LLM models using LangChain
    
    Supports:
    - OpenAI models (gpt-3.5-turbo, gpt-4, gpt-4o, etc.)
    - Custom API endpoints (via base_url)
    - Google Gemini (optional, requires langchain-google-genai)
    """
    
    def __init__(self, 
                 model_name: str,
                 temp: float = 0., 
                 top_p: float = 0.,
                 n_out: int = 1, 
                 max_new_tokens: int = 256, 
                 seed=None, 
                 stop=None,
                 args=None):
        """Initialize LLM wrapper
        
        Args:
            model_name: Model name or path
            temp: Temperature for generation
            top_p: Top-p sampling parameter
            n_out: Number of outputs to generate
            max_new_tokens: Maximum new tokens
            seed: Random seed
            stop: Stop sequences
            args: Additional arguments (e.g., base_url)
        """
        
        self.args = args

        self.temp = temp
        self.topp = top_p
        self.n = n_out
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.stop = stop

        # Usage tracking
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_cost = 0

        # Initialize model
        if "gpt" in model_name:
            self.model = self.openai(model_name)
        
        elif args and hasattr(args, 'base_url') and args.base_url != "":
            self.model = self.hf_openai(model_name)

        elif "gemini" in model_name:
            self.model = self.gemini(model_name)

        else:
            raise NotImplementedError(f"Model {model_name} not supported")
        

    def get_usage(self):
        """Get token usage and cost
        
        Returns:
            Dictionary with completion_tokens, prompt_tokens, and cost
        """
        return {
            "completion_tokens": self.completion_tokens, 
            "prompt_tokens": self.prompt_tokens, 
            "cost": self.total_cost
        }

    def openai(self, model_name):
        """Initialize OpenAI model
        
        Args:
            model_name: OpenAI model name
            
        Returns:
            ChatOpenAI instance
        """
        if model_name != "gpt-3.5-turbo":
            print(f"[WARNING] Current model is {model_name}, not 'gpt-3.5-turbo'.")
        self.model_name = model_name

        return ChatOpenAI(
            model_name=model_name,
            n=self.n,
            temperature=self.temp,
        )

    def hf_openai(self, model_name):
        """Initialize model with custom API endpoint (e.g., vLLM, HuggingFace)
        
        Args:
            model_name: Model name
            
        Returns:
            ChatOpenAI instance with custom base_url
        """
        self.model_name = model_name

        return ChatOpenAI(  
            openai_api_base=self.args.base_url,
            model_name=model_name,
            n=self.n,
            temperature=self.temp,
            model_kwargs={
                "stop": self.stop,
                "seed": self.seed,
            }
        )

    def gemini(self, model_name):
        """Initialize Google Gemini model
        
        Args:
            model_name: Gemini model name
            
        Returns:
            ChatGoogleGenerativeAI instance
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Please install langchain-google-genai: pip install langchain-google-genai")
        
        self.model_name = model_name

        return ChatGoogleGenerativeAI(
            model=model_name,
            convert_system_message_to_human=True,  # Gemini does not support system messages
            n=self.n,
            temperature=self.temp,
            top_p=self.topp,
            stop=self.stop,
            top_k=None
        )

