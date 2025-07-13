from dotenv import load_dotenv
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = load_dotenv("HF_TOKEN", default=None)
    # openai_api_url: str = "https://api.openai.com/v1/chat/completions"
    openai_api_url: str = "https://router.huggingface.co/v1/chat/completions"
    # anonymizer_salt: str = "change-me-in-production"
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()