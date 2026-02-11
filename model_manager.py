"""
模型管理器 - 懒加载单例模式
首次处理文档时才加载模型，之后复用
"""

import os
import sys
import threading
import logging
from config_manager import get_settings

logger = logging.getLogger(__name__)

# 确保模型目录在 Python 路径中
MODEL_DIR = os.environ.get("DEEPSEEK_OCR_MODEL_PATH", "./model_weights")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)


class ModelManager:
    """线程安全的单例模型管理器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.model = None
                    cls._instance.tokenizer = None
                    cls._instance._loading = False
                    cls._instance._loaded = False
                    cls._instance._error = None
                    cls._instance._load_lock = threading.Lock()
        return cls._instance

    def get_status(self):
        if self._loaded:
            return "loaded"
        elif self._loading:
            return "loading"
        elif self._error:
            return "error"
        else:
            return "not_loaded"

    def get_status_detail(self):
        return {
            "status": self.get_status(),
            "error": self._error,
        }

    def ensure_loaded(self):
        """确保模型已加载，如果未加载则同步加载（阻塞）"""
        if self._loaded:
            return True

        with self._load_lock:
            # Double-check after acquiring lock
            if self._loaded:
                return True
            if self._error:
                raise RuntimeError(f"Model failed to load previously: {self._error}")

            self._loading = True
            try:
                self._load_model()
                return True
            except Exception as e:
                self._error = str(e)
                self._loading = False
                logger.exception("Failed to load model")
                raise

    def _load_model(self):
        import torch

        settings = get_settings()
        model_path = settings.model_path.strip()
        if not model_path:
            model_path = os.environ.get("DEEPSEEK_OCR_MODEL_PATH", MODEL_DIR)

        # 确保模型路径在 Python 路径中
        if model_path not in sys.path:
            sys.path.insert(0, model_path)

        logger.info(f"Loading tokenizer from {model_path}...")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        logger.info(f"Loading model from {model_path}...")
        from modeling_deepseekocr2 import DeepseekOCR2ForCausalLM

        self.model = DeepseekOCR2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )

        self.model = self.model.to(torch.bfloat16).cpu()
        self.model.eval()
        self.model.disable_torch_init()

        self._loaded = True
        self._loading = False
        logger.info("Model loaded successfully")


# 全局单例
model_manager = ModelManager()
