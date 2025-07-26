# main.py
import os
from utils.config import Config

def main():
    """Main entry point for the application."""
    # Set environment (default to local)
    env = os.getenv("ENVIRONMENT", "local")
    config = Config(environment=env)

    print(f"Running in {config.environment} environment")
    print(f"Debug mode: {config.debug}")

    if config.debug:
        # Show the config object for debugging
        print(f"Config: {config}")

if __name__ == "__main__":
    main()