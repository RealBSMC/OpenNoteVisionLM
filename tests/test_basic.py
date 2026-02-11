"""
Basic tests for OpenNoteVision LM
Run with: pytest tests/test_basic.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that core modules can be imported"""
    # Test config imports
    import config
    assert hasattr(config, 'MODEL_PATH')
    assert hasattr(config, 'IMAGE_SIZE')
    
    # Test manager imports (without loading model)
    import config_manager
    import model_manager
    
    # Test engine imports
    import ocr_engine
    import rag_engine
    
    # Ensure required functions exist
    assert hasattr(ocr_engine, 'get_pdf_page_count')
    assert hasattr(rag_engine, 'RAGConfig')
    
    print("All imports successful")


def test_config_manager():
    """Test configuration manager"""
    from config_manager import get_settings, RAGSettings
    
    settings = get_settings()
    assert isinstance(settings, RAGSettings)
    
    # Check default values
    assert settings.provider == "openai"
    assert settings.summary_token_threshold == 200
    assert settings.context_length == 8192
    
    print("Config manager test passed")


def test_model_manager():
    """Test model manager initialization"""
    from model_manager import ModelManager
    
    manager = ModelManager()
    assert manager.get_status() in ["not_loaded", "loading", "loaded", "error"]
    
    # Test status detail structure
    status_detail = manager.get_status_detail()
    assert "status" in status_detail
    assert "error" in status_detail
    
    print("Model manager test passed")


def test_environment_variables():
    """Check that required environment variables are documented"""
    # This test doesn't enforce values, just checks documentation
    required_env_vars = [
        "DEEPSEEK_OCR_MODEL_PATH",
        "OPENAI_API_KEY",  # or alternatives
    ]
    
    env_example_path = project_root / ".env.example"
    assert env_example_path.exists(), ".env.example file missing"
    
    env_content = env_example_path.read_text()
    for var in required_env_vars:
        assert var in env_content, f"{var} not documented in .env.example"
    
    print("Environment variables documentation check passed")


if __name__ == "__main__":
    print("Running basic tests...")
    test_imports()
    test_config_manager()
    test_model_manager()
    test_environment_variables()
    print("\nAll basic tests passed!")