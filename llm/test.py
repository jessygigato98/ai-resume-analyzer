
from huggingface_client import HuggingFaceLLMClient

client = HuggingFaceLLMClient()

prompt = """
Extract technical skills from the text below.
Return JSON only.

Text:
Python, Django, REST APIs, PostgreSQL, Git
"""

print(client.generate(prompt))
