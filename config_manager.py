"""
配置管理模块 - 支持多厂商 LLM 配置
"""

import os
import json
import logging
import threading
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "data" / "config"
CONFIG_FILE = CONFIG_DIR / "rag_config.json"

LLMProvider = Literal["openai", "openrouter", "deepseek", "custom"]


class RAGSettings(BaseModel):
    provider: LLMProvider = "openai"
    api_key: str = ""
    api_base_url: str = ""
    model_path: str = ""
    index_model: str = "gpt-4o-2024-11-20"
    chat_model: str = "gpt-4o-2024-11-20"
    rag_model: str = ""
    summary_token_threshold: int = 200
    context_length: int = 8192
    enable_rag: bool = True
    
    def is_configured(self) -> bool:
        return bool(self.api_key.strip())
    
    def get_api_base_url(self) -> str:
        if self.api_base_url:
            return self.api_base_url
        return DEFAULT_BASE_URLS.get(self.provider, "https://api.openai.com/v1")
    
    def get_index_model(self) -> str:
        if self.rag_model and not self.index_model:
            return self.rag_model
        return self.index_model
    
    def get_chat_model(self) -> str:
        if self.rag_model and not self.chat_model:
            return self.rag_model
        return self.chat_model
    
    def to_display(self) -> dict:
        masked_key = ""
        if self.api_key:
            if len(self.api_key) > 8:
                masked_key = self.api_key[:4] + "..." + self.api_key[-4:]
            else:
                masked_key = "***"
        
        return {
            "provider": self.provider,
            "api_key": masked_key,
            "api_base_url": self.api_base_url or DEFAULT_BASE_URLS.get(self.provider, ""),
            "model_path": self.model_path,
            "index_model": self.get_index_model(),
            "chat_model": self.get_chat_model(),
            "rag_model": self.rag_model,
            "summary_token_threshold": self.summary_token_threshold,
            "context_length": self.context_length,
            "enable_rag": self.enable_rag,
            "is_configured": self.is_configured(),
        }


DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "custom": "",
}

MODELS_BY_PROVIDER = {
    "openai": [
        {"value": "gpt-4o-2024-11-20", "label": "GPT-4o (推荐)", "description": "最新最强多模态模型"},
        {"value": "gpt-4o-mini", "label": "GPT-4o Mini", "description": "更快更便宜"},
        {"value": "gpt-4-turbo", "label": "GPT-4 Turbo", "description": "经典 GPT-4"},
        {"value": "gpt-4", "label": "GPT-4", "description": "原始 GPT-4"},
        {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo", "description": "经济实惠"},
    ],
    "openrouter": [
        {"value": "openai/gpt-4o", "label": "GPT-4o (via OpenRouter)", "description": "OpenAI 最新"},
        {"value": "anthropic/claude-3.5-sonnet", "label": "Claude 3.5 Sonnet", "description": "Anthropic 最强"},
        {"value": "anthropic/claude-3-opus", "label": "Claude 3 Opus", "description": "Anthropic 旗舰"},
        {"value": "google/gemini-pro-1.5", "label": "Gemini Pro 1.5", "description": "Google 大模型"},
        {"value": "meta-llama/llama-3.1-70b-instruct", "label": "Llama 3.1 70B", "description": "Meta 开源"},
        {"value": "deepseek/deepseek-chat", "label": "DeepSeek Chat", "description": "DeepSeek 对话"},
        {"value": "qwen/qwen-2.5-72b-instruct", "label": "Qwen 2.5 72B", "description": "阿里通义"},
    ],
    "deepseek": [
        {"value": "deepseek-chat", "label": "DeepSeek Chat (推荐)", "description": "通用对话模型"},
        {"value": "deepseek-reasoner", "label": "DeepSeek Reasoner", "description": "推理增强模型（R1）"},
    ],
    "custom": [],
}


def ensure_config_dir():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> RAGSettings:
    settings = RAGSettings()
    
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if data.get("provider"):
                settings.provider = data["provider"]
            if data.get("api_key"):
                settings.api_key = data["api_key"]
            if data.get("api_base_url"):
                settings.api_base_url = data["api_base_url"]
            if data.get("index_model"):
                settings.index_model = data["index_model"]
            if data.get("chat_model"):
                settings.chat_model = data["chat_model"]
            if data.get("rag_model"):
                settings.rag_model = data["rag_model"]
                if not settings.index_model:
                    settings.index_model = settings.rag_model
                if not settings.chat_model:
                    settings.chat_model = settings.rag_model
            if data.get("summary_token_threshold"):
                settings.summary_token_threshold = data["summary_token_threshold"]
            if data.get("context_length"):
                settings.context_length = data["context_length"]
            if "enable_rag" in data:
                settings.enable_rag = data["enable_rag"]
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")
    
    if not settings.api_key:
        env_key = (
            os.getenv("OPENAI_API_KEY") or 
            os.getenv("CHATGPT_API_KEY") or
            os.getenv("OPENROUTER_API_KEY") or
            os.getenv("DEEPSEEK_API_KEY")
        )
        if env_key:
            settings.api_key = env_key
            if os.getenv("OPENROUTER_API_KEY"):
                settings.provider = "openrouter"
            elif os.getenv("DEEPSEEK_API_KEY"):
                settings.provider = "deepseek"
    
    return settings


def save_settings(settings: RAGSettings):
    ensure_config_dir()
    data = {
        "provider": settings.provider,
        "api_key": settings.api_key,
        "api_base_url": settings.api_base_url,
        "index_model": settings.index_model,
        "chat_model": settings.chat_model,
        "rag_model": settings.rag_model,
        "summary_token_threshold": settings.summary_token_threshold,
        "context_length": settings.context_length,
        "enable_rag": settings.enable_rag,
    }
    CONFIG_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("RAG settings saved")


def update_settings(**kwargs) -> RAGSettings:
    global _settings_cache
    with _settings_lock:
        settings = load_settings()
        
        old_provider = settings.provider
        new_provider = kwargs.get('provider')
        
        for key, value in kwargs.items():
            if hasattr(settings, key) and value is not None:
                setattr(settings, key, value)
        
        if new_provider and new_provider != old_provider:
            if new_provider != 'custom':
                settings.api_base_url = ""
            default_models = MODELS_BY_PROVIDER.get(new_provider, [])
            if default_models and not settings.index_model:
                settings.index_model = default_models[0]["value"]
            if default_models and not settings.chat_model:
                settings.chat_model = default_models[0]["value"]
        
        _settings_cache = None
        save_settings(settings)
        logger.info(f"Settings updated: provider={settings.provider}, index_model={settings.index_model}, chat_model={settings.chat_model}, base_url={settings.api_base_url or '(default)'}")
        return settings


_settings_cache: Optional[RAGSettings] = None
_settings_lock = threading.Lock()

def get_settings() -> RAGSettings:
    """获取配置（带缓存，线程安全）"""
    global _settings_cache
    if _settings_cache is None:
        with _settings_lock:
            # Double-check after acquiring lock
            if _settings_cache is None:
                _settings_cache = load_settings()
                logger.info(f"Settings loaded from file: provider={_settings_cache.provider}, model={_settings_cache.rag_model}, base_url={_settings_cache.api_base_url}")
    return _settings_cache


def refresh_settings() -> RAGSettings:
    """刷新配置缓存"""
    global _settings_cache
    with _settings_lock:
        _settings_cache = load_settings()
        return _settings_cache


def clear_settings_cache():
    """清除配置缓存"""
    global _settings_cache
    with _settings_lock:
        _settings_cache = None


def get_models_for_provider(provider: str) -> list:
    """获取指定提供商的模型列表"""
    return MODELS_BY_PROVIDER.get(provider, [])
