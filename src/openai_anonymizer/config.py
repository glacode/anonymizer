import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    load_dotenv()
    openai_api_key: str = os.environ.get("HF_TOKEN")
    # openai_api_url: str = "https://api.openai.com/v1/chat/completions"
    openai_api_url: str = "https://router.huggingface.co/v1/chat/completions"
    # anonymizer_salt: str = "change-me-in-production"
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # class Config:
    #     env_file = ".env"

settings = Settings()