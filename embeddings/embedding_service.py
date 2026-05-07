from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
import dotenv

api_key = dotenv.get_key(".env", "HUGGINGFACEHUB_API_TOKEN")
print('api_key',api_key)

def generate_embedding(text: str) -> list[float]:
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=api_key
    )
    
    query_result = embeddings.embed_query(text)
    return query_result