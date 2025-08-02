# main.py
import os
import argparse
from utils.config import Config

def main_cli():
    """CLI entry point for the application."""
    env = os.getenv("ENVIRONMENT", "local")
    config = Config(environment=env)
    
    print(f"Running in {config.environment} environment")
    print(f"Debug mode: {config.debug}")
    
    if config.debug:
        print(f"Config: {config}")

def main_api():
    """FastAPI entry point for the application."""
    try:
        from api.app import create_app
        import uvicorn
    except ImportError as e:
        print(f"Error importing API dependencies: {e}")
        print("Make sure FastAPI and uvicorn are installed")
        return
    
    env = os.getenv("ENVIRONMENT", "local")
    config = Config(environment=env, project_name="sk_mcp_demo")
    
    app = create_app(config)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=config.debug
    )

def main():
    """Main entry point - determines mode based on arguments."""
    parser = argparse.ArgumentParser(description="SK MCP Demo")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli", 
                       help="Run mode: cli or api")
    parser.add_argument("--env", default="local",
                       help="Environment (affects which .env file to load)")
    
    args = parser.parse_args()
    
    # Set environment before loading config
    os.environ["ENVIRONMENT"] = args.env
    
    if args.mode == "api":
        main_api()
    else:
        main_cli()

if __name__ == "__main__":
    main()