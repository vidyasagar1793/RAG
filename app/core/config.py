from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Metadata
    PROJECT_NAME: str = "Enterprise RAG API"
    VERSION: str = "0.1.0"

    # PostgreSQL Config (from your docker-compose)
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "localhost" 
    POSTGRES_PORT: int = 5432
    
    # Qdrant Config
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # LLM Provider
    OPENAI_API_KEY: str

    @property
    def postgres_uri(self) -> str:
        """Dynamically build the connection string."""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Tell Pydantic to look for a .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Instantiate it once to be imported across your app
settings = Settings()