"""
OCR 引擎 - 从现有脚本提取的核心处理逻辑
提供 PDF 转图片、逐页 OCR、后处理、RAG 索引构建等功能
"""

import os
import gc
import io
import re
import ast
import uuid
import tempfile
import logging
import asyncio
from pathlib import Path
from typing import Optional, Callable
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# PDF 处理
# ============================================================

def pdf_to_images(pdf_path, dpi=144):
    """将 PDF 转为 PIL Image 列表"""
    import fitz

    images = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            Image.MAX_IMAGE_PIXELS = None
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)

    return images


def get_pdf_page_count(pdf_path):
    """获取 PDF 页数"""
    import fitz

    with fitz.open(pdf_path) as pdf_doc:
        return pdf_doc.page_count


def render_pdf_page_to_png(pdf_path, page_num, dpi=144):
    """将 PDF 指定页渲染为 PNG 字节"""
    import fitz

    with fitz.open(pdf_path) as pdf_doc:
        if page_num >= pdf_doc.page_count:
            return None

        page = pdf_doc[page_num]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")


# ============================================================
# 结果解析与后处理
# ============================================================

def re_match(text):
    """解析检测标签，分离图片和其他引用"""
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for a_match in matches:
        if "<|ref|>image<|/ref|>" in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


def extract_and_save_images(output_text, page_idx, images_dir, original_image):
    """从检测结果中提取图片区域并保存"""
    _, matches_image, _ = re_match(output_text)
    image_width, image_height = original_image.size

    coord_pattern = r"<\|ref\|>.*?<\|/ref\|><\|det\|>(.*?)<\|/det\|>"

    for idx, a_match_image in enumerate(matches_image):
        try:
            m = re.search(coord_pattern, a_match_image, re.DOTALL)
            if m:
                cor_list = ast.literal_eval(m.group(1))
                for points in cor_list:
                    x1, y1, x2, y2 = points
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)
                    cropped = original_image.crop((x1, y1, x2, y2))
                    save_path = os.path.join(images_dir, f"{page_idx}_{idx}.jpg")
                    cropped.save(save_path, quality=95)
        except Exception as e:
            logger.warning(f"Failed to extract image {page_idx}_{idx}: {e}")


def postprocess_page_output(output_text, page_idx):
    """将原始 OCR 输出后处理为干净的 Markdown"""
    content = output_text

    # 解析引用标签
    _, matches_image, matches_other = re_match(content)

    # 将图片引用标签替换为 Markdown 图片语法
    for idx, a_match_image in enumerate(matches_image):
        content = content.replace(
            a_match_image,
            f"![](images/{page_idx}_{idx}.jpg)\n",
        )

    # 移除其他引用标签并清理
    for a_match_other in matches_other:
        content = (
            content.replace(a_match_other, "")
            .replace("\\coloneqq", ":=")
            .replace("\\eqqcolon", "=:")
            .replace("\n\n\n\n", "\n\n")
            .replace("\n\n\n", "\n\n")
        )

    return content.strip()


# ============================================================
# 核心处理流程
# ============================================================

def _save_image_to_temp(image, temp_dir):
    """保存 PIL Image 到临时文件"""
    temp_file = os.path.join(temp_dir, f"{uuid.uuid4().hex}.jpg")
    image.save(temp_file, format="JPEG", quality=95)
    return temp_file


def process_pdf_document(pdf_path, doc_dir, model, tokenizer, on_page_done=None):
    """
    处理整个 PDF 文档，逐页执行 OCR 并保存结果。

    Args:
        pdf_path: PDF 文件路径
        doc_dir: 文档存储目录
        model: 已加载的 OCR 模型
        tokenizer: 分词器
        on_page_done: 每页完成时的回调 fn(page_idx, total_pages)

    Returns:
        总页数
    """
    from config import PROMPT, CROP_MODE, IMAGE_SIZE, BASE_SIZE

    pages_dir = os.path.join(doc_dir, "pages")
    images_dir = os.path.join(doc_dir, "images")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # PDF 转图片
    logger.info(f"Converting PDF to images: {pdf_path}")
    pil_images = pdf_to_images(pdf_path)
    total_pages = len(pil_images)
    logger.info(f"Total pages: {total_pages}")

    with tempfile.TemporaryDirectory() as temp_dir:
        for page_idx, pil_image in enumerate(pil_images):
            logger.info(f"Processing page {page_idx + 1}/{total_pages}")

            # 保存图片到临时文件供模型读取
            temp_image_path = _save_image_to_temp(pil_image, temp_dir)

            try:
                # 运行 OCR 推理
                raw_output = model.infer(
                    tokenizer=tokenizer,
                    prompt=PROMPT,
                    image_file=temp_image_path,
                    output_path=images_dir,
                    base_size=BASE_SIZE,
                    image_size=IMAGE_SIZE,
                    crop_mode=CROP_MODE,
                    eval_mode=True,
                    save_results=False,
                )

                # 保存检测输出（含坐标标签）
                det_path = os.path.join(pages_dir, f"page_{page_idx}_det.mmd")
                with open(det_path, "w", encoding="utf-8") as f:
                    f.write(raw_output)

                # 从检测结果中提取并保存图片
                extract_and_save_images(raw_output, page_idx, images_dir, pil_image)

                # 后处理为干净的 Markdown
                clean_content = postprocess_page_output(raw_output, page_idx)

                # 保存干净的 Markdown
                mmd_path = os.path.join(pages_dir, f"page_{page_idx}.mmd")
                with open(mmd_path, "w", encoding="utf-8") as f:
                    f.write(clean_content)

                logger.info(f"Page {page_idx + 1} done, output length: {len(clean_content)}")

            except Exception as e:
                logger.exception(f"Error processing page {page_idx}")
                # 保存错误标记
                mmd_path = os.path.join(pages_dir, f"page_{page_idx}.mmd")
                with open(mmd_path, "w", encoding="utf-8") as f:
                    f.write(f"> **OCR Error on this page:** {str(e)}")

            # 清理内存
            gc.collect()

            # 回调
            if on_page_done:
                on_page_done(page_idx, total_pages)

    return total_pages


# ============================================================
# RAG 索引构建
# ============================================================

def combine_pages_to_markdown(pages_dir: str) -> str:
    """将所有页面的 Markdown 合并为一个完整文档"""
    pages_path = Path(pages_dir)
    if not pages_path.exists():
        raise FileNotFoundError(f"Pages directory not found: {pages_dir}")
    
    # 只匹配 page_N.mmd，排除 page_N_det.mmd（检测结果原始文件）
    md_files = sorted(
        (f for f in pages_path.glob("page_*.mmd") if re.match(r'^page_\d+$', f.stem)),
        key=lambda p: int(p.stem.split("_")[1])
    )
    
    combined = []
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        combined.append(content)
    
    return "\n\n---\n\n".join(combined)


async def build_rag_index(
    doc_dir: str,
    doc_id: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    构建文档的 RAG 索引（PageIndex 树结构）
    
    Args:
        doc_dir: 文档目录路径
        doc_id: 文档 ID
        on_progress: 进度回调函数
    
    Returns:
        树结构字典
    """
    from pathlib import Path
    import sys
    
    PAGEINDEX_DIR = Path(__file__).parent / "PageIndex"
    if str(PAGEINDEX_DIR) not in sys.path:
        sys.path.insert(0, str(PAGEINDEX_DIR))
    
    from pageindex.page_index_md import md_to_tree
    
    pages_dir = os.path.join(doc_dir, "pages")
    index_path = os.path.join(doc_dir, "index.json")
    
    if on_progress:
        on_progress("combining_pages")
    
    markdown_content = combine_pages_to_markdown(pages_dir)
    
    if on_progress:
        on_progress("building_tree")
    
    temp_md_path = f"/tmp/{doc_id}_temp.md"
    with open(temp_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    try:
        tree_structure = await md_to_tree(
            md_path=temp_md_path,
            if_thinning=False,
            if_add_node_summary="yes",
            summary_token_threshold=200,
            model="gpt-4o-2024-11-20",
            if_add_doc_description="no",
            if_add_node_text="no",
            if_add_node_id="yes",
        )
        
        with open(index_path, "w", encoding="utf-8") as f:
            import json
            json.dump(tree_structure, f, ensure_ascii=False, indent=2)
        
        logger.info(f"RAG index built for document {doc_id}")
        return tree_structure
        
    finally:
        if os.path.exists(temp_md_path):
            os.remove(temp_md_path)


def build_rag_index_sync(
    doc_dir: str,
    doc_id: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """同步版本的索引构建"""
    return asyncio.run(build_rag_index(doc_dir, doc_id, on_progress))
