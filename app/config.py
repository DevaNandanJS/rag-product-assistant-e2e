from pydantic_settings import BaseSettings, SettingsConfigDict
class settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra= "ignore")

    # secrets 
    gemini_api_key= ""
    groq_api_key= ""

    # RAG Parameters 
    chunk_size: int= 500
    chunk_overlap: int= 75
    top_k: int= 5
    score_threshold: float= 0.3

settings= settings()
