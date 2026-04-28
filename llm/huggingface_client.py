import requests
import dotenv
from huggingface_hub import InferenceClient
from typer import prompt

token = dotenv.get_key(".env", "HUGGINGFACEHUB_API_TOKEN")

class HuggingFaceLLMClient:
    """
    Wrapper around Hugging Face InferenceClient
    for text generation / instruction following.
    """

    def __init__(
        self,
        # model: str = "CohereLabs/tiny-aya-earth"
        # model: str = "mistralai/Mistral-7B-Instruct-v0.2"
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        timeout: int = 60
    ):
        self.api_token =token
        if not self.api_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set")

        self.client = InferenceClient(
            model=model,
            token=token,
            provider="together",
            timeout=timeout
        )

    def generate(self, prompt: str) -> str:  
        """
        Generate text from a prompt using Hugging Face Inference API.
        """       
        response = self.client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            return_full_text=False
        )

        # text_generation can return str or list[str] depending on backend
        if isinstance(response, list):
            return response[0]

        return response