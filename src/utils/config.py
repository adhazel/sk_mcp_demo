import os
from pathlib import Path
from dotenv import load_dotenv

def load_environment(env: str = "local"):
    """Load environment variables based on environment."""
    
    # Find the project root (where pyproject.toml is located)
    current_dir = Path(__file__).parent
    project_root = current_dir
    
    # Walk up the directory tree to find pyproject.toml
    while project_root.parent != project_root:
        if (project_root / "pyproject.toml").exists():
            break
        project_root = project_root.parent
    
    # Load environment-specific file from project root
    env_file = project_root / f".env.{env}"
    if env_file.exists():
        load_dotenv(env_file, override=True)
        print(f"✅ Loaded environment file: {env_file}")
    else:
        print(f"⚠️  Environment file not found: {env_file}")
        print(f"💡 Expected location: {env_file.absolute()}")

def _get_valid_env_value(key: str, default: str = None) -> str:
    """
    Get environment variable value, but return None if it contains 'placeholder'.
    
    Args:
        key: Environment variable key
        default: Default value to return if key is not found or contains placeholder
        
    Returns:
        Environment variable value or default if value contains 'placeholder'
    """
    value = os.getenv(key, default)
    
    # Return None if value contains 'placeholder' (case-insensitive)
    if value and 'placeholder' in value.lower():
        return None
    
    return value

class Config:
    def __init__(self, environment: str = "local"):
        load_environment(environment)
        
        # Store the environment
        self.environment = environment
        
        # OpenAI Configuration
        self.openai_api_key = _get_valid_env_value("OPENAI_API_KEY")
        self.openai_model = _get_valid_env_value("OPENAI_MODEL")

        # Azure OpenAI
        self.openai_api_type = _get_valid_env_value("OPENAI_API_TYPE", "openai")
        self.azure_openai_endpoint = _get_valid_env_value("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_key = _get_valid_env_value("AZURE_OPENAI_API_KEY")
        self.azure_openai_model = _get_valid_env_value("AZURE_OPENAI_MODEL")
        self.azure_openai_deployment = _get_valid_env_value("AZURE_OPENAI_DEPLOYMENT")
        self.azure_openai_api_version = _get_valid_env_value("AZURE_OPENAI_API_VERSION")

        # Azure OpenAI Embedding
        self.azure_openai_embedding_endpoint = _get_valid_env_value("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        self.azure_openai_embedding_api_key = _get_valid_env_value("AZURE_OPENAI_EMBEDDING_API_KEY")
        self.azure_openai_embedding_model = _get_valid_env_value("AZURE_OPENAI_EMBEDDING_MODEL")
        self.azure_openai_embedding_deployment = _get_valid_env_value("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.azure_openai_embedding_api_version = _get_valid_env_value("AZURE_OPENAI_EMBEDDING_API_VERSION")

        # Vector Store Configuration
        self.chroma_db_path = _get_valid_env_value("CHROMA_DB_PATH", f"./data/chroma_db")
        self.vector_collection_name = _get_valid_env_value("VECTOR_COLLECTION_NAME", "products")

        # Web Search Configuration (for your RAG + web search feature)
        self.search_api_key = _get_valid_env_value("SEARCH_API_KEY")
        self.search_engine_id = _get_valid_env_value("SEARCH_ENGINE_ID")

        # Application Settings
        self.log_level = _get_valid_env_value("LOG_LEVEL", "INFO")
        self.debug = _get_valid_env_value("DEBUG", "false").lower() == "true"

        # Semantic Kernel specific settings
        self.sk_log_level = _get_valid_env_value("SK_LOG_LEVEL", self.log_level)

        # Validate required settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required configuration is present."""
        required_settings = {}

        # add OPENAI_API_KEY to required_settings if OPENAI_API_TYPE is not set or not "azure"
        if self.openai_api_type.lower() != "azure":
            required_settings["OPENAI_API_KEY"] = self.openai_api_key
        else:
            required_settings["AZURE_OPENAI_API_KEY"] = self.azure_openai_api_key
            required_settings["OPENAI_API_TYPE"] = self.openai_api_type
        
        missing = [key for key, value in required_settings.items() if not value]
        
        if missing:
            print(f"❌ Missing required environment variables: {', '.join(missing)}")
            print(f"📝 Create .env.{self.environment} file with these variables")
        else:
            print(f"✅ All required configuration loaded for '{self.environment}' environment")
    
    def __repr__(self):
        """String representation for debugging."""
        return (
            f"Config("
            f"environment={self.environment!r}, "
            f"openai_api_key={'***' if self.openai_api_key else None}, "
            f"openai_model={self.openai_model!r}, "
            f"openai_api_type={self.openai_api_type!r}, "
            f"azure_openai_endpoint={self.azure_openai_endpoint!r}, "
            f"azure_openai_api_key={'***' if self.azure_openai_api_key else None}, "
            f"azure_openai_model={self.azure_openai_model!r}, "
            f"azure_openai_deployment={self.azure_openai_deployment!r}, "
            f"azure_openai_api_version={self.azure_openai_api_version!r}, "
            f"azure_openai_embedding_endpoint={self.azure_openai_embedding_endpoint!r}, "
            f"azure_openai_embedding_api_key={'***' if self.azure_openai_embedding_api_key else None}, "
            f"azure_openai_embedding_model={self.azure_openai_embedding_model!r}, "
            f"azure_openai_embedding_deployment={self.azure_openai_embedding_deployment!r}, "
            f"azure_openai_embedding_api_version={self.azure_openai_embedding_api_version!r}, "
            f"chroma_db_path={self.chroma_db_path!r}, "
            f"vector_collection_name={self.vector_collection_name!r}, "
            f"search_api_key={'***' if self.search_api_key else None}, "
            f"search_engine_id={self.search_engine_id!r}, "
            f"log_level={self.log_level!r}, "
            f"debug={self.debug}, "
            f"sk_log_level={self.sk_log_level!r}"
            f")"
        )
    
    def to_dict(self) -> dict:
        """
        Return configuration as a dictionary for easier access in notebooks.
        Excludes None values and internal attributes.
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and value is not None:
                config_dict[key] = value
        return config_dict

# import os
# from pathlib import Path
# from dotenv import load_dotenv

# def load_environment(env: str = "local"):
#     """Load environment variables based on environment."""
    
#     # Load environment-specific file
#     env_file = Path(f".env.{env}")
#     if env_file.exists():
#         load_dotenv(env_file)
#         print(f"✅ Loaded environment file: {env_file}")
#     else:
#         print(f"⚠️  Environment file not found: {env_file}")

# class Config:
#     def __init__(self, environment: str = "local"):
#         load_environment(environment)
        
#         # Store the environment
#         self.environment = environment
        
#         # OpenAI Configuration
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
#         # Vector Store Configuration
#         self.chroma_db_path = os.getenv("CHROMA_DB_PATH", f"./data/{environment}_chroma_db")
#         self.vector_collection_name = os.getenv("VECTOR_COLLECTION_NAME", "products")
        
#         # Web Search Configuration (for your RAG + web search feature)
#         self.search_api_key = os.getenv("SEARCH_API_KEY")
#         self.search_engine_id = os.getenv("SEARCH_ENGINE_ID")
        
#         # Application Settings
#         self.log_level = os.getenv("LOG_LEVEL", "INFO")
#         self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
#         # Semantic Kernel specific settings
#         self.sk_log_level = os.getenv("SK_LOG_LEVEL", self.log_level)
        
#         # Validate required settings
#         self._validate_config()
    
#     def _validate_config(self):
#         """Validate that required configuration is present."""
#         required_settings = {
#             "OPENAI_API_KEY": self.openai_api_key,
#         }
        
#         missing = [key for key, value in required_settings.items() if not value]
        
#         if missing:
#             print(f"❌ Missing required environment variables: {', '.join(missing)}")
#             print(f"📝 Create .env.{self.environment} file with these variables")
#         else:
#             print(f"✅ All required configuration loaded for '{self.environment}' environment")
    
#     def __repr__(self):
#         """String representation for debugging."""
#         # return f"Config(environment='{self.environment}', model='{self.openai_model}', debug={self.debug})"
#         return f"Config(environment='{self.environment}', debug={self.debug})"
    