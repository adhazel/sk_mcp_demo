"""
Notebook utilities for the mcp_rag project.

This module provides helper functions to set up the environment
for Jupyter notebooks in this project.
"""

import sys
import os
from pathlib import Path
from typing import Optional

def setup_notebook_environment(project_name: str = "mcp_rag") -> Path:
    """
    Set up the notebook environment by adding the project root to Python path.
    
    Args:
        project_name: The name of the project directory to look for
        
    Returns:
        Path to the project root directory
        
    Raises:
        FileNotFoundError: If the project root cannot be found
    """
    # Start from current directory and walk up to find project root
    current_path = Path().resolve()
    
    # Look for project root by finding the directory with pyproject.toml
    # or the directory name matching project_name
    for path in [current_path] + list(current_path.parents):
        if (path / "pyproject.toml").exists() or path.name == project_name:
            project_root = path
            break
    else:
        # Fallback: assume notebooks are one level down from project root
        project_root = current_path.parent
    
    # Add project root to Python path if not already present
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"✅ Added to Python path: {project_root}")
    else:
        print(f"✅ Already in Python path: {project_root}")
    
    return project_root

def load_config(environment: str = "local"):
    """
    Load the project configuration.
    
    Args:
        environment: The environment to load (default: "local")
        
    Returns:
        Config object with loaded environment variables
    """
    try:
        from src.utils.config import Config
        config = Config(environment=environment)
        print(f"✅ Configuration loaded for '{environment}' environment")
        return config
    except ImportError as e:
        print(f"❌ Failed to import config: {e}")
        print("Make sure you've run setup_notebook_environment() first")
        raise
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        raise

def notebook_setup(environment: str = "local", project_name: str = "mcp_rag"):
    """
    Complete notebook setup - adds project to path and loads config.
    
    Args:
        environment: The environment to load (default: "local")
        project_name: The name of the project directory
        
    Returns:
        tuple: (project_root_path, config_object)
    """
    print("🚀 Setting up notebook environment...")
    
    # Setup project path
    project_root = setup_notebook_environment(project_name)
    
    # Load configuration
    config = load_config(environment)
    
    print("✨ Notebook setup complete!")
    print(f"   Project root: {project_root}")
    print(f"   Environment: {environment}")
    print(f"   Config: {config}")
    
    return project_root, config
