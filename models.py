"""
LLM model wrappers using LangChain
Supports OpenAI GPT models, HuggingFace models via OpenAI API, and Google Gemini
"""

from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI


class LLMs(object):
    """Wrapper class for different LLM providers using LangChain"""
    
    def __init__(self, 
                 model_name: str,
                 temp: float = 0., 
                 top_p: float = 0.,
                 n_out: int = 1, 
                 max_new_tokens: int = 256, 
                 seed=None, 
                 stop=None,
                 base_url: str = "",
                 args=None):
        
        self.args = args
        self.temp = temp
        self.topp = top_p
        self.n = n_out
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.stop = stop
        self.base_url = base_url
        self.model_name = model_name

        # Usage tracking
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_cost = 0

        # Initialize the appropriate model
        if "gpt" in model_name:
            self.model = self._init_openai(model_name)
        elif base_url != "":
            self.model = self._init_hf_openai(model_name)
        elif "gemini" in model_name:
            self.model = self._init_gemini(model_name)
        else:
            raise NotImplementedError(f"Model {model_name} not supported")
        
    def get_usage(self):
        """Get current token usage and cost"""
        return {
            "completion_tokens": self.completion_tokens, 
            "prompt_tokens": self.prompt_tokens, 
            "cost": self.total_cost
        }

    def _init_openai(self, model_name):
        """Initialize OpenAI model (GPT-3.5, GPT-4, etc.)"""
        if model_name != "gpt-3.5-turbo":
            print(f"[WARNING] Current model is {model_name}")
        
        return ChatOpenAI(
            model_name=model_name,
            n=self.n,
            temperature=self.temp,
            max_tokens=self.max_new_tokens
        )

    def _init_hf_openai(self, model_name):
        """Initialize HuggingFace model via OpenAI-compatible API"""
        return ChatOpenAI(  
            openai_api_base=self.base_url,
            model_name=model_name,
            n=self.n,
            temperature=self.temp,
            max_tokens=self.max_new_tokens,
            model_kwargs={
                "stop": self.stop,
                "seed": self.seed,
            }
        )

    def _init_gemini(self, model_name):
        """Initialize Google Gemini model"""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            convert_system_message_to_human=True,  # Gemini doesn't support system messages
            n=self.n,
            temperature=self.temp,
            top_p=self.topp,
            stop=self.stop,
            top_k=None
        )

