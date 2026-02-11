"""
DeepSeek OCR 2 - Web Application
FastAPI 后端 + 静态前端
"""

# ====== CPU 优化（必须在 import torch 之前） ======
import os
import sys

_PHYSICAL_CORES = (os.cpu_count() or 16) // 2
os.environ["OMP_NUM_THREADS"] = str(_PHYSICAL_CORES)
os.environ["MKL_NUM_THREADS"] = str(_PHYSICAL_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(_PHYSICAL_CORES)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 确保模型目录在 Python 路径中
MODEL_DIR = os.environ.get("DEEPSEEK_OCR_MODEL_PATH", "./model_weights")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

# ====== 标准库导入 ======
import uuid
import json
import shutil
import logging
import threading
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

# ====== 第三方库导入 ======
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ====== 本地模块 ======
from model_manager import model_manager
from ocr_engine import process_pdf_document, get_pdf_page_count, render_pdf_page_to_png
from rag_engine import rag_engine
from config_manager import get_settings, update_settings, refresh_settings, get_models_for_provider

# ====== 配置 ======
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "documents"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ====== FastAPI App ======
app = FastAPI(title="DeepSeek OCR 2", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def add_no_cache_for_static(request, call_next):
    """开发阶段：禁止静态文件缓存"""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response

# 追踪后台处理任务
_processing_tasks: dict[str, threading.Thread] = {}


# ============================================================
# 请求模型
# ============================================================

class ChatRequest(BaseModel):
    query: str


class BuildIndexRequest(BaseModel):
    pass


class SettingsUpdate(BaseModel):
    provider: Optional[str] = None
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    model_path: Optional[str] = None
    index_model: Optional[str] = None
    chat_model: Optional[str] = None
    rag_model: Optional[str] = None
    summary_token_threshold: Optional[int] = None
    context_length: Optional[int] = None
    enable_rag: Optional[bool] = None


# ============================================================
# 工具函数
# ============================================================

def _read_metadata(doc_id: str) -> dict:
    metadata_path = DATA_DIR / doc_id / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(404, "Document not found")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _write_metadata(doc_id: str, metadata: dict):
    metadata_path = DATA_DIR / doc_id / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _update_metadata(doc_id: str, **kwargs):
    metadata = _read_metadata(doc_id)
    metadata.update(kwargs)
    _write_metadata(doc_id, metadata)


# ============================================================
# 后台处理
# ============================================================

def _doc_exists(doc_id: str) -> bool:
    """检查文档目录是否仍然存在"""
    return (DATA_DIR / doc_id / "metadata.json").exists()


def _background_process(doc_id: str):
    """后台线程：加载模型 → 逐页 OCR → 更新状态"""
    try:
        if not _doc_exists(doc_id):
            logger.warning(f"[{doc_id}] Document was deleted before processing started")
            return
        
        # 加载模型（懒加载，首次会较慢）
        _update_metadata(doc_id, status="loading_model")
        logger.info(f"[{doc_id}] Ensuring model is loaded...")
        model_manager.ensure_loaded()

        if not _doc_exists(doc_id):
            logger.warning(f"[{doc_id}] Document was deleted during model loading")
            return

        # 开始处理
        _update_metadata(doc_id, status="processing")
        logger.info(f"[{doc_id}] Starting OCR processing...")
        start_time = time.time()

        doc_dir = DATA_DIR / doc_id

        def on_page_done(page_idx, total_pages):
            if not _doc_exists(doc_id):
                raise RuntimeError(f"Document {doc_id} was deleted during processing")
            _update_metadata(doc_id, processed_pages=page_idx + 1)
            logger.info(f"[{doc_id}] Page {page_idx + 1}/{total_pages} completed")

        process_pdf_document(
            pdf_path=str(doc_dir / "original.pdf"),
            doc_dir=str(doc_dir),
            model=model_manager.model,
            tokenizer=model_manager.tokenizer,
            on_page_done=on_page_done,
        )

        elapsed = time.time() - start_time
        if _doc_exists(doc_id):
            _update_metadata(
                doc_id, status="completed", processing_time=round(elapsed, 1)
            )
        logger.info(f"[{doc_id}] Processing completed in {elapsed:.1f}s")

    except Exception as e:
        logger.exception(f"[{doc_id}] Processing failed")
        try:
            if _doc_exists(doc_id):
                _update_metadata(doc_id, status="error", error=str(e))
        except Exception:
            pass
    finally:
        _processing_tasks.pop(doc_id, None)


# ============================================================
# API 路由
# ============================================================

@app.get("/")
async def index():
    """主页面"""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """上传 PDF 文件，返回文档 ID 并启动后台处理"""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    # 生成文档 ID
    doc_id = uuid.uuid4().hex[:8]
    doc_dir = DATA_DIR / doc_id
    doc_dir.mkdir(parents=True)
    (doc_dir / "pages").mkdir()
    (doc_dir / "images").mkdir()

    # 保存 PDF
    pdf_path = doc_dir / "original.pdf"
    content = await file.read()
    pdf_path.write_bytes(content)

    # 获取页数
    page_count = get_pdf_page_count(str(pdf_path))

    # 创建元数据
    metadata = {
        "id": doc_id,
        "filename": file.filename,
        "page_count": page_count,
        "processed_pages": 0,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "processing_time": None,
        "error": None,
    }
    _write_metadata(doc_id, metadata)

    # 启动后台处理线程
    thread = threading.Thread(
        target=_background_process, args=(doc_id,), daemon=True
    )
    thread.start()
    _processing_tasks[doc_id] = thread

    logger.info(f"Uploaded '{file.filename}' as {doc_id} ({page_count} pages)")
    return metadata


@app.get("/api/documents")
async def list_documents():
    """列出所有文档"""
    docs = []
    if DATA_DIR.exists():
        for doc_dir in sorted(DATA_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            metadata_path = doc_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    docs.append(json.loads(metadata_path.read_text(encoding="utf-8")))
                except Exception:
                    pass
    return docs


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """获取单个文档元数据"""
    return _read_metadata(doc_id)


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档及所有结果"""
    doc_dir = DATA_DIR / doc_id
    if not doc_dir.exists():
        raise HTTPException(404, "Document not found")
    
    # 如果文档正在处理中，先标记为已删除，让后台线程感知
    if doc_id in _processing_tasks:
        logger.warning(f"[{doc_id}] Deleting document that is still being processed")
        _processing_tasks.pop(doc_id, None)
    
    shutil.rmtree(str(doc_dir), ignore_errors=True)
    logger.info(f"Deleted document {doc_id}")
    return {"status": "deleted", "id": doc_id}


@app.get("/api/documents/{doc_id}/page/{page_num}/content")
async def get_page_content(doc_id: str, page_num: int):
    """获取指定页的 .mmd 内容"""
    mmd_path = DATA_DIR / doc_id / "pages" / f"page_{page_num}.mmd"
    if not mmd_path.exists():
        raise HTTPException(404, "Page not found or not yet processed")
    return {
        "content": mmd_path.read_text(encoding="utf-8"),
        "page": page_num,
    }


@app.get("/api/documents/{doc_id}/page/{page_num}/pdf-image")
async def get_page_pdf_image(doc_id: str, page_num: int):
    """渲染 PDF 指定页为 PNG 图片"""
    pdf_path = DATA_DIR / doc_id / "original.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "Document not found")

    img_bytes = render_pdf_page_to_png(str(pdf_path), page_num)
    if img_bytes is None:
        raise HTTPException(404, "Page number out of range")

    return Response(content=img_bytes, media_type="image/png")


@app.get("/api/documents/{doc_id}/pdf")
async def get_pdf(doc_id: str):
    """获取原始 PDF 文件"""
    pdf_path = DATA_DIR / doc_id / "original.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "Document not found")
    return FileResponse(str(pdf_path), media_type="application/pdf")


@app.get("/api/documents/{doc_id}/images/{filename}")
async def get_extracted_image(doc_id: str, filename: str):
    """获取提取的图片"""
    # 安全检查：防止路径遍历
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "Invalid filename")
    img_path = DATA_DIR / doc_id / "images" / filename
    if not img_path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(img_path))


@app.get("/api/model/status")
async def get_model_status():
    """获取模型加载状态"""
    return model_manager.get_status_detail()


# ============================================================
# 配置管理 API
# ============================================================

@app.get("/api/settings")
async def get_rag_settings():
    """获取 RAG 配置"""
    settings = get_settings()
    return settings.to_display()


@app.get("/api/providers")
async def get_providers():
    """获取支持的 LLM 提供商列表"""
    from config_manager import MODELS_BY_PROVIDER, DEFAULT_BASE_URLS
    return {
        "providers": [
            {"value": "openai", "label": "OpenAI", "default_base_url": DEFAULT_BASE_URLS["openai"]},
            {"value": "openrouter", "label": "OpenRouter", "default_base_url": DEFAULT_BASE_URLS["openrouter"]},
            {"value": "deepseek", "label": "DeepSeek", "default_base_url": DEFAULT_BASE_URLS["deepseek"]},
            {"value": "custom", "label": "自定义", "default_base_url": ""},
        ],
        "models_by_provider": MODELS_BY_PROVIDER,
    }


@app.get("/api/providers/{provider}/models")
async def get_provider_models(provider: str):
    """获取指定提供商的模型列表"""
    models = get_models_for_provider(provider)
    return {"provider": provider, "models": models}


@app.post("/api/settings")
async def update_rag_settings(request: SettingsUpdate):
    update_data = {}
    if request.provider is not None:
        update_data["provider"] = request.provider
    if request.api_key is not None:
        update_data["api_key"] = request.api_key
    if request.api_base_url is not None:
        update_data["api_base_url"] = request.api_base_url
    if request.model_path is not None:
        update_data["model_path"] = request.model_path
    if request.index_model is not None:
        update_data["index_model"] = request.index_model
    if request.chat_model is not None:
        update_data["chat_model"] = request.chat_model
    if request.rag_model is not None:
        update_data["rag_model"] = request.rag_model
    if request.summary_token_threshold is not None:
        update_data["summary_token_threshold"] = request.summary_token_threshold
    if request.context_length is not None:
        update_data["context_length"] = request.context_length
    if request.enable_rag is not None:
        update_data["enable_rag"] = request.enable_rag
    
    if update_data:
        new_settings = update_settings(**update_data)
        rag_engine.update_config(new_settings)
        return {"status": "success", "settings": new_settings.to_display()}
    return {"status": "no_changes"}


# ============================================================
# RAG API 路由
# ============================================================

@app.post("/api/documents/{doc_id}/build-index")
async def build_document_index(doc_id: str, background_tasks: BackgroundTasks):
    """为文档构建 RAG 索引"""
    settings = get_settings()
    if not settings.is_configured():
        raise HTTPException(400, "请先在设置中配置 API Key")
    
    doc_dir = DATA_DIR / doc_id
    if not doc_dir.exists():
        raise HTTPException(404, "Document not found")
    
    metadata = _read_metadata(doc_id)
    if metadata.get("status") != "completed":
        raise HTTPException(400, "Document OCR not completed yet")
    
    index_path = doc_dir / "index.json"
    if index_path.exists():
        return {"status": "already_exists", "message": "Index already built"}
    
    def build_index_task():
        try:
            _update_metadata(doc_id, index_status="building")
            
            # 确保使用最新配置
            current_settings = get_settings()
            rag_engine.update_config(current_settings)
            
            # 使用 rag_engine 构建索引
            import asyncio
            tree = asyncio.run(rag_engine.build_index(
                doc_id,
                str(doc_dir / "pages"),
                str(index_path)
            ))
            
            _update_metadata(doc_id, index_status="completed")
            logger.info(f"[{doc_id}] Index built successfully")
        except Exception as e:
            logger.exception(f"[{doc_id}] Index build failed")
            _update_metadata(doc_id, index_status="error", index_error=str(e))
    
    background_tasks.add_task(build_index_task)
    return {"status": "started", "message": "Index building started"}


@app.get("/api/documents/{doc_id}/index-status")
async def get_index_status(doc_id: str):
    """获取文档索引状态"""
    doc_dir = DATA_DIR / doc_id
    index_path = doc_dir / "index.json"
    
    if not index_path.exists():
        metadata = _read_metadata(doc_id)
        if metadata.get("index_status") != "not_built":
            _update_metadata(doc_id, index_status="not_built", index_error=None)
        return {"index_status": "not_built", "index_error": None}
    
    metadata = _read_metadata(doc_id)
    return {
        "index_status": metadata.get("index_status", "completed"),
        "index_error": metadata.get("index_error"),
    }


@app.get("/api/documents/{doc_id}/tree")
async def get_document_tree(doc_id: str):
    """获取文档的树结构索引"""
    doc_dir = DATA_DIR / doc_id
    index_path = doc_dir / "index.json"
    
    if not index_path.exists():
        raise HTTPException(404, "Index not found. Build index first.")
    
    tree = rag_engine.get_tree(doc_id)
    if tree is None:
        tree = json.loads(index_path.read_text(encoding="utf-8"))
        pages_dir = doc_dir / "pages"
        rag_engine.load_document(doc_id, str(index_path), str(pages_dir))
    
    return tree


@app.post("/api/documents/{doc_id}/chat")
async def chat_with_document(doc_id: str, request: ChatRequest):
    """与文档进行对话（RAG 问答）"""
    doc_dir = DATA_DIR / doc_id
    index_path = doc_dir / "index.json"
    
    if not index_path.exists():
        raise HTTPException(404, "Index not found. Build index first.")
    
    # 每次聊天都刷新配置，确保使用最新设置
    settings = get_settings()
    if not settings.is_configured():
        raise HTTPException(500, "请先在设置中配置 API Key")
    
    rag_engine.update_config(settings)
    
    if not rag_engine.is_loaded(doc_id):
        pages_dir = doc_dir / "pages"
        rag_engine.load_document(doc_id, str(index_path), str(pages_dir))
    
    try:
        result = await rag_engine.ask(doc_id, request.query)
        result["config_used"] = {
            "provider": settings.provider,
            "chat_model": settings.get_chat_model(),
            "base_url": settings.get_api_base_url(),
        }
        return result
    except ValueError as e:
        if "API Key" in str(e):
            raise HTTPException(500, "API Key not configured. Set API Key in settings.")
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.exception(f"Chat error for document {doc_id}")
        raise HTTPException(500, f"Chat failed: {str(e)}")


@app.post("/api/documents/{doc_id}/search")
async def search_document(doc_id: str, request: ChatRequest):
    """在文档中搜索相关内容"""
    doc_dir = DATA_DIR / doc_id
    index_path = doc_dir / "index.json"
    
    if not index_path.exists():
        raise HTTPException(404, "Index not found. Build index first.")
    
    if not rag_engine.is_loaded(doc_id):
        pages_dir = doc_dir / "pages"
        rag_engine.load_document(doc_id, str(index_path), str(pages_dir))
    
    try:
        result = await rag_engine.search(doc_id, request.query)
        return result
    except ValueError as e:
        if "OPENAI_API_KEY" in str(e):
            raise HTTPException(500, "OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.exception(f"Search error for document {doc_id}")
        raise HTTPException(500, f"Search failed: {str(e)}")


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    logger.info(f"Starting DeepSeek OCR 2 Web App")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"CPU cores: {_PHYSICAL_CORES}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
