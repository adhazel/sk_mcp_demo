import os
from pathlib import Path
from dotenv import load_dotenv

def load_environment(env: str = "local"):
    """Load environment variables based on environment."""
    
    # Load environment-specific file
    env_file = Path(f".env.{env}")
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment file: {env_file}")
    else:
        print(f"⚠️  Environment file not found: {env_file}")

class Config:
    def __init__(self, environment: str = "local"):
        load_environment(environment)
        
        # Store the environment
        self.environment = environment
        
        # OpenAI Configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # Vector Store Configuration
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", f"./data/{environment}_chroma_db")
        self.vector_collection_name = os.getenv("VECTOR_COLLECTION_NAME", "products")
        
        # Web Search Configuration (for your RAG + web search feature)
        self.search_api_key = os.getenv("SEARCH_API_KEY")
        self.search_engine_id = os.getenv("SEARCH_ENGINE_ID")
        
        # Application Settings
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Semantic Kernel specific settings
        self.sk_log_level = os.getenv("SK_LOG_LEVEL", self.log_level)
        
        # Validate required settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required configuration is present."""
        required_settings = {
            "OPENAI_API_KEY": self.openai_api_key,
        }
        
        missing = [key for key, value in required_settings.items() if not value]
        
        if missing:
            print(f"❌ Missing required environment variables: {', '.join(missing)}")
            print(f"📝 Create .env.{self.environment} file with these variables")
        else:
            print(f"✅ All required configuration loaded for '{self.environment}' environment")
    
    def __repr__(self):
        """String representation for debugging."""
        # return f"Config(environment='{self.environment}', model='{self.openai_model}', debug={self.debug})"
        return f"Config(environment='{self.environment}', debug={self.debug})"
    