import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()  # loads .env from current working directory

def get_client():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not api_key or not endpoint:
        raise RuntimeError("Missing AZURE_OPENAI_* values in .env")

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )
