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
        model: str = "CohereLabs/tiny-aya-earth",
        #model: str = "mistralai/Mixtral-8x22B-Instruct-v0.1",
        # model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        # model: str = "CohereLabs/command-a-reasoning-08-2025", #provider="cohere",
        timeout: int = 60
    ):
        self.api_token =token
        if not self.api_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set")

        self.client = InferenceClient(
            model=model,
            token=token,
            # provider="nscale",
            provider="cohere",
            timeout=timeout
        )

    def generate(self, prompt: str) -> str:  
        """
        Generate text from a prompt using Hugging Face Inference API.
        """       
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7
        )
        print("response",response.choices[0].message.content)
        return response.choices[0].message.content