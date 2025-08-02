"""
NAME: config.py
DESCRIPTION: Configuration management with the following features:
- Loads environment variables from a .env.[<environment>] file
- Adds the project root to the environment
- Replaces 'placeholder' values with None, allows for default values
- Saves named configuration values as attributes
- Provides a Config class to access configuration values.
- Provides a method to convert configuration to a dictionary

AUTHOR: April Hazel
CREDIT: Derived from: 
    https://github.com/modelcontextprotocol/python-sdk/blob/959d4e39ae13e45d3059ec6d6ca82fb231039a91/examples/servers/simple-streamablehttp/mcp_simple_streamablehttp/event_store.py
HISTORY:
    - 20240730: Initial implementation
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from .caller import get_caller
import logging
from  openai import AzureOpenAI, OpenAI

def load_environment(env: str = "local", project_name: str = "mcp_rag"):
    """Load environment variables based on environment."""

    logging.debug(f"🐛 DEBUG: Config file being executed: {__file__}")

    # Get the project root 
    caller_path = get_caller()
    current_path = Path(__file__).parent
     # Walk up the directory tree to find a folder named project_name
    project_root = current_path
    while project_root.parent != project_root:  # Stop at filesystem root
        if project_root.name == project_name:
            break
        project_root = project_root.parent
    else:
        # If we reached the filesystem root without finding project_name,
        # use the original fallback logic
        if caller_path and 'notebooks' in str(caller_path):
            project_root = caller_path.parent.parent
        else:
            project_root = Path(__file__).parent.parent.parent

    # Set PROJECT_ROOT as an environment variable
    os.environ["PROJECT_ROOT"] = str(project_root)
    
    # Load environment-specific file from project root
    env_file = project_root / f".env.{env}"
    
    if env_file.exists():
        load_dotenv(env_file, override=True)
        logging.debug(f"✅ Loaded environment file: {env_file}")
    else:
        logging.warning(f"⚠️  Environment file not found: {env_file}")
        logging.warning(f"💡 Expected location: {env_file.absolute()}")


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
    if value and 'placeholder' in value.lower():
        return None
    return value

class Config:
    
    def __init__(self, environment: str = "local", project_name: str = "sk_mcp_demo"):
        try: 
            # Configure basic logging first with a default level
            logging.basicConfig(
                level=logging.INFO,  # Default level for initial setup
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                force=True
            )

            # Load environment (now logging will work in load_environment)
            load_environment(environment, project_name=project_name)
            logging.info(f"✅ Loaded environment file for '{environment}' environment")

            self.log_level = _get_valid_env_value("LOG_LEVEL", "INFO").upper()

            # Reconfigure logging with the final level from environment
            final_level = getattr(logging, self.log_level, logging.INFO)
            logging.basicConfig(
                level=final_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                force=True
            )
            logging.info(f"🔧 Logging configured at level: {self.log_level}")
            
        except Exception as e:
            logging.error(f"❌ Failed to load environment variables: {e}")
            raise
        
        # Store the environment
        self.environment = environment
        
        # Get project root from environment variable
        self.project_root = Path(_get_valid_env_value("PROJECT_ROOT"))
        
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
        self.chroma_db_path = _get_valid_env_value("CHROMA_DB_PATH", "./data/chroma_db")

        # Web Search Configuration (for your RAG + web search feature)
        self.serp_api_key = _get_valid_env_value("SERP_API_KEY")

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
            logging.error(f"❌ Missing required environment variables: {', '.join(missing)}")
            logging.error(f"📝 Create .env.{self.environment} file with these variables")
        else:
            logging.info(f"✅ All required configuration loaded for '{self.environment}' environment")
    
    def __repr__(self):
        """String representation for debugging."""
        return (
            f"Config("
            f"log_level={self.log_level!r}, "
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
            f"serp_api_key={'***' if self.serp_api_key else None}, "
            f"project_root={self.project_root!r}"
            f")"
        )
    
    def get_llm(self):
        """
        Get the configured LLM client based on the environment.
        
        Returns:
            An instance of the configured LLM client.
        """
        if self.openai_api_type.lower() == "azure":
            return AzureOpenAI(
                api_key=self.azure_openai_api_key,
                endpoint=self.azure_openai_endpoint,
                model=self.azure_openai_model,
                deployment=self.azure_openai_deployment,
                api_version=self.azure_openai_api_version
            )
        else:
            return OpenAI(api_key=self.openai_api_key, model=self.openai_model)

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
