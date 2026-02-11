"""
RAG 引擎 - 基于 PageIndex 的推理式检索
无向量、基于树结构搜索的 RAG 系统
支持多厂商 LLM (OpenAI, OpenRouter, DeepSeek, 自定义)
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional

import openai

logger = logging.getLogger(__name__)

PAGEINDEX_DIR = Path(__file__).parent / "PageIndex"
if str(PAGEINDEX_DIR) not in sys.path:
    sys.path.insert(0, str(PAGEINDEX_DIR))

from pageindex.page_index_md import md_to_tree
from pageindex.utils import count_tokens


class RAGConfig:
    api_key: str = ""
    api_base_url: str = "https://api.openai.com/v1"
    index_model: str = "gpt-4o-2024-11-20"
    chat_model: str = "gpt-4o-2024-11-20"
    summary_token_threshold: int = 200
    context_length: int = 8192
    provider: str = "openai"


def combine_pages_to_markdown(pages_dir: str) -> str:
    """将所有页面的 Markdown 合并为一个完整文档"""
    import re as _re
    pages_path = Path(pages_dir)
    if not pages_path.exists():
        raise FileNotFoundError(f"Pages directory not found: {pages_dir}")
    
    # 只匹配 page_N.mmd，排除 page_N_det.mmd（检测结果原始文件）
    md_files = sorted(
        (f for f in pages_path.glob("page_*.mmd") if _re.match(r'^page_\d+$', f.stem)),
        key=lambda p: int(p.stem.split("_")[1])
    )
    
    combined = []
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        combined.append(content)
    
    return "\n\n---\n\n".join(combined)


def _get_openai_client(config: RAGConfig) -> openai.AsyncOpenAI:
    """创建 OpenAI 客户端，支持自定义 base_url"""
    return openai.AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.api_base_url,
        timeout=60.0,  # 超时 60 秒
    )


async def _api_call_with_retry(client, model, messages, temperature=0, extra_headers=None, max_retries=5):
    """带指数退避重试的 API 调用，专门处理 429 错误"""
    for i in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                extra_headers=extra_headers,
            )
            return response
        except openai.RateLimitError:
            wait_time = min(2 ** i + 1, 60)
            logger.warning(f"Rate limited (429), retry {i+1}/{max_retries}, waiting {wait_time}s...")
            if i < max_retries - 1:
                await asyncio.sleep(wait_time)
            else:
                raise
        except openai.APITimeoutError:
            wait_time = min(2 ** i + 1, 30)
            logger.warning(f"API timeout, retry {i+1}/{max_retries}, waiting {wait_time}s...")
            if i < max_retries - 1:
                await asyncio.sleep(wait_time)
            else:
                raise


async def build_tree_index_from_markdown(
    markdown_content: str,
    doc_name: str = "document",
    config: Optional[RAGConfig] = None,
) -> dict:
    """
    从 Markdown 内容构建 PageIndex 树索引
    
    Args:
        markdown_content: Markdown 文本内容
        doc_name: 文档名称
        config: RAG 配置
    
    Returns:
        包含树结构的字典
    """
    if config is None:
        config = RAGConfig()
    
    if not config.api_key:
        raise ValueError("API Key is required")
    
    temp_md_path = Path(f"/tmp/{doc_name}_temp.md")
    temp_md_path.write_text(markdown_content, encoding="utf-8")
    
    # 仅保存需要修改的环境变量的原始值
    _env_keys_to_restore = ["CHATGPT_API_KEY", "CHATGPT_BASE_URL"]
    original_values = {k: os.environ.get(k) for k in _env_keys_to_restore}
    
    try:
        # 设置 PageIndex 需要的环境变量
        os.environ["CHATGPT_API_KEY"] = config.api_key
        if config.api_base_url:
            os.environ["CHATGPT_BASE_URL"] = config.api_base_url
        elif "CHATGPT_BASE_URL" in os.environ:
            del os.environ["CHATGPT_BASE_URL"]
        
        # 更新 PageIndex utils 模块的全局变量
        import pageindex.utils as utils_module
        utils_module.CHATGPT_API_KEY = config.api_key
        utils_module.CHATGPT_BASE_URL = config.api_base_url or None
        
        tree_structure = await md_to_tree(
            md_path=str(temp_md_path),
            if_thinning=False,
            if_add_node_summary="yes",
            summary_token_threshold=config.summary_token_threshold,
            model=config.index_model,
            if_add_doc_description="no",
            if_add_node_text="yes",
            if_add_node_id="yes",
            context_length=config.context_length,
        )
        return tree_structure
    finally:
        # 仅恢复修改过的环境变量，而不是清除所有环境变量
        for key, orig_val in original_values.items():
            if orig_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = orig_val
        if temp_md_path.exists():
            temp_md_path.unlink()


def save_tree_index(tree_structure: dict, output_path: str):
    """保存树索引到文件"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tree_structure, f, ensure_ascii=False, indent=2)
    logger.info(f"Tree index saved to: {output_path}")


def load_tree_index(index_path: str) -> dict:
    """加载树索引"""
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def tree_search(
    query: str,
    tree_structure: dict,
    page_contents: dict[int, str],
    config: Optional[RAGConfig] = None,
) -> dict:
    """
    基于树结构的推理式检索
    
    Args:
        query: 用户问题
        tree_structure: PageIndex 树结构
        page_contents: 页码 -> 页面内容的映射
        config: RAG 配置
    
    Returns:
        包含检索结果和相关内容的字典
    """
    if config is None:
        config = RAGConfig()
    
    if not config.api_key:
        raise ValueError("API Key is required for tree search")
    
    structure = tree_structure.get("structure", tree_structure)
    
    def format_tree_for_prompt(nodes, indent=0) -> str:
        lines = []
        for node in nodes:
            prefix = "  " * indent
            title = node.get("title", "Untitled")
            node_id = node.get("node_id", "")
            summary = node.get("summary", "")
            
            line = f"{prefix}- [{node_id}] {title}"
            if summary:
                line += f": {summary[:100]}..."
            lines.append(line)
            
            if node.get("nodes"):
                lines.append(format_tree_for_prompt(node["nodes"], indent + 1))
        return "\n".join(lines)
    
    tree_text = format_tree_for_prompt(structure if isinstance(structure, list) else [structure])
    
    search_prompt = f"""你是一个文档检索专家。用户有一个问题，你需要根据文档的目录结构，找出最相关的章节。

文档目录结构:
{tree_text}

用户问题: {query}

请分析问题，找出最相关的章节。返回 JSON 格式:
{{
    "thinking": "<分析过程>",
    "relevant_nodes": ["node_id1", "node_id2"],
    "reasoning": "<为什么选择这些章节>"
}}

只返回 JSON，不要其他内容。"""

    client = _get_openai_client(config)
    
    extra_headers = None
    if config.provider == "openrouter":
        extra_headers = {
            "HTTP-Referer": "https://deepseek-ocr.local",
            "X-Title": "DeepSeek OCR RAG",
        }
    
    response = await _api_call_with_retry(
        client,
        model=config.chat_model,
        messages=[{"role": "user", "content": search_prompt}],
        temperature=0,
        extra_headers=extra_headers,
    )
    
    result_text = response.choices[0].message.content
    
    try:
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        search_result = json.loads(result_text.strip())
    except json.JSONDecodeError:
        search_result = {"thinking": result_text, "relevant_nodes": [], "reasoning": ""}
    
    def find_nodes_by_ids(nodes, target_ids: list[str]) -> list[dict]:
        found = []
        for node in nodes:
            if node.get("node_id") in target_ids:
                found.append(node)
            if node.get("nodes"):
                found.extend(find_nodes_by_ids(node["nodes"], target_ids))
        return found
    
    relevant_nodes = find_nodes_by_ids(
        structure if isinstance(structure, list) else [structure],
        search_result.get("relevant_nodes", [])
    )
    
    return {
        "query": query,
        "search_result": search_result,
        "relevant_nodes": relevant_nodes,
    }


async def rag_answer(
    query: str,
    tree_structure: dict,
    page_contents: dict[int, str],
    config: Optional[RAGConfig] = None,
) -> dict:
    """
    RAG 问答：检索 + 生成答案
    
    Args:
        query: 用户问题
        tree_structure: PageIndex 树结构
        page_contents: 页码 -> 页面内容
        config: RAG 配置
    
    Returns:
        包含答案和来源的字典
    """
    if config is None:
        config = RAGConfig()
    
    search_result = await tree_search(query, tree_structure, page_contents, config)
    
    context_parts = []
    for node in search_result.get("relevant_nodes", []):
        node_id = node.get("node_id", "")
        title = node.get("title", "")
        summary = node.get("summary", "")
        node_text = node.get("text", "")
        
        content_text = node_text if node_text else summary
        context_parts.append(f"### {title} (node_id: {node_id})\n摘要: {summary}\n内容:\n{content_text}")
    
    context = "\n\n---\n\n".join(context_parts) if context_parts else "未找到相关内容"
    
    answer_prompt = f"""你是一个专业的文档问答助手。根据提供的文档内容回答用户问题。

文档内容:
{context}

用户问题: {query}

请基于文档内容回答问题。如果文档中没有相关信息，请明确说明。
回答时请标注信息来源（如：根据第X页...）。

直接回答问题，不要其他内容。"""

    client = _get_openai_client(config)
    
    extra_headers = None
    if config.provider == "openrouter":
        extra_headers = {
            "HTTP-Referer": "https://deepseek-ocr.local",
            "X-Title": "DeepSeek OCR RAG",
        }
    
    response = await _api_call_with_retry(
        client,
        model=config.chat_model,
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0,
        extra_headers=extra_headers,
    )
    
    answer = response.choices[0].message.content
    usage = response.usage
    
    return {
        "query": query,
        "answer": answer,
        "sources": [node.get("title") for node in search_result.get("relevant_nodes", [])],
        "search_thinking": search_result.get("search_result", {}).get("thinking", ""),
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        },
    }


class RAGEngine:
    """RAG 引擎封装类"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._tree_cache: dict[str, dict] = {}
        self._page_cache: dict[str, dict[int, str]] = {}
    
    def update_config(self, settings):
        self.config.api_key = settings.api_key
        self.config.index_model = settings.get_index_model()
        self.config.chat_model = settings.get_chat_model()
        self.config.api_base_url = settings.get_api_base_url()
        self.config.provider = settings.provider
        self.config.summary_token_threshold = settings.summary_token_threshold
        self.config.context_length = settings.context_length
        logger.info(f"RAG config updated: provider={self.config.provider}, index_model={self.config.index_model}, chat_model={self.config.chat_model}, base_url={self.config.api_base_url}, context_length={self.config.context_length}")
    
    def load_document(self, doc_id: str, index_path: str, pages_dir: str):
        """加载文档索引和页面内容"""
        self._tree_cache[doc_id] = load_tree_index(index_path)
        
        pages_path = Path(pages_dir)
        self._page_cache[doc_id] = {}
        
        for md_file in pages_path.glob("page_*.mmd"):
            page_num = int(md_file.stem.split("_")[1])
            self._page_cache[doc_id][page_num] = md_file.read_text(encoding="utf-8")
    
    async def build_index(self, doc_id: str, pages_dir: str, output_path: str) -> dict:
        """构建文档索引"""
        markdown_content = combine_pages_to_markdown(pages_dir)
        tree_structure = await build_tree_index_from_markdown(
            markdown_content, 
            doc_name=doc_id,
            config=self.config
        )
        save_tree_index(tree_structure, output_path)
        
        self._tree_cache[doc_id] = tree_structure
        self._page_cache[doc_id] = {}
        pages_path = Path(pages_dir)
        for md_file in pages_path.glob("page_*.mmd"):
            page_num = int(md_file.stem.split("_")[1])
            self._page_cache[doc_id][page_num] = md_file.read_text(encoding="utf-8")
        
        return tree_structure
    
    async def search(self, doc_id: str, query: str) -> dict:
        """检索相关内容"""
        if doc_id not in self._tree_cache:
            raise ValueError(f"Document {doc_id} not loaded")
        
        return await tree_search(
            query,
            self._tree_cache[doc_id],
            self._page_cache[doc_id],
            self.config
        )
    
    async def ask(self, doc_id: str, query: str) -> dict:
        """问答"""
        if doc_id not in self._tree_cache:
            raise ValueError(f"Document {doc_id} not loaded")
        
        return await rag_answer(
            query,
            self._tree_cache[doc_id],
            self._page_cache[doc_id],
            self.config
        )
    
    def get_tree(self, doc_id: str) -> Optional[dict]:
        """获取文档树结构"""
        return self._tree_cache.get(doc_id)
    
    def is_loaded(self, doc_id: str) -> bool:
        """检查文档是否已加载"""
        return doc_id in self._tree_cache


rag_engine = RAGEngine()
