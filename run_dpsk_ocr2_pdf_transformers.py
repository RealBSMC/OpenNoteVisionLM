import os
import sys
import gc
import time
import tempfile
from pathlib import Path

# ====== CPU 优化：线程配置（必须在 import torch 之前设置） ======
# AMD Ryzen 7 8845HS: 8物理核心 / 16逻辑线程
# 对 AVX512 密集计算，只用物理核心数最优（SMT 会争抢执行单元和缓存）
_PHYSICAL_CORES = (os.cpu_count() or 16) // 2  # 8 物理核心
os.environ["OMP_NUM_THREADS"] = str(_PHYSICAL_CORES)
os.environ["MKL_NUM_THREADS"] = str(_PHYSICAL_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(_PHYSICAL_CORES)
# 强制CPU模式
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

# 设置 PyTorch 线程数 = 物理核心数
torch.set_num_threads(_PHYSICAL_CORES)

# 添加模型目录到Python路径以允许直接导入
model_dir = os.environ.get("DEEPSEEK_OCR_MODEL_PATH", "./model_weights")
sys.path.insert(0, model_dir)

print(f"PyTorch version: {torch.__version__}")
print(f"Device: cpu ({_PHYSICAL_CORES} physical cores)")
print(f"CPU threads (intra-op): {torch.get_num_threads()}")
print(f"CPU capability: {torch.backends.cpu.get_cpu_capability()}")
print(f"MKL enabled: {torch.backends.mkl.is_available()}")

# 导入原始脚本的内容
import fitz
import img2pdf
import io
import re
import ast
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from config import (
    MODEL_PATH,
    INPUT_PATH,
    OUTPUT_PATH,
    PROMPT,
    SKIP_REPEAT,
    MAX_CONCURRENCY,
    NUM_WORKERS,
    CROP_MODE,
    IMAGE_SIZE,
    BASE_SIZE,
)

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 导入Transformers替代vLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor


class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """
    pdf2images
    """
    images = []

    pdf_document = fitz.open(pdf_path)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(
                    img, mask=img.split()[-1] if img.mode == "RGBA" else None
                )
                img = background

        images.append(img)

    pdf_document.close()
    return images


def pil_to_pdf_img2pdf(pil_images, output_path):
    if not pil_images:
        return

    image_bytes_list = []

    for img in pil_images:
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)

    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    except Exception as e:
        print(f"error: {e}")


def re_match(text):
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if "<|ref|>image<|/ref|>" in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = ast.literal_eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, jdx):
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (
                    np.random.randint(0, 200),
                    np.random.randint(0, 200),
                    np.random.randint(0, 255),
                )

                color_a = color + (20,)
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == "image":
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{OUTPUT_PATH}/images/{jdx}_{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1

                    try:
                        if label_type == "title":
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle(
                                [x1, y1, x2, y2],
                                fill=color_a,
                                outline=(0, 0, 0, 0),
                                width=1,
                            )
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle(
                                [x1, y1, x2, y2],
                                fill=color_a,
                                outline=(0, 0, 0, 0),
                                width=1,
                            )

                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle(
                            [
                                text_x,
                                text_y,
                                text_x + text_width,
                                text_y + text_height,
                            ],
                            fill=(255, 255, 255, 30),
                        )

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except Exception:
                        pass
        except Exception:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, jdx):
    result_image = draw_bounding_boxes(image, ref_texts, jdx)
    return result_image


def process_single_image(image, temp_dir):
    """保存图像到临时文件并返回路径"""
    import uuid

    temp_file = os.path.join(temp_dir, f"{uuid.uuid4().hex}.jpg")
    image.save(temp_file, format="JPEG", quality=95)
    return temp_file


if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/images", exist_ok=True)

    print(f"{Colors.RED}PDF loading .....{Colors.RESET}")

    images = pdf_to_images_high_quality(INPUT_PATH)

    print(f"{Colors.GREEN}Loaded {len(images)} pages{Colors.RESET}")

    # 加载模型和分词器
    print(f"{Colors.YELLOW}Loading model from {MODEL_PATH}...{Colors.RESET}")

    # 直接导入模型类（已修改相对导入）
    from modeling_deepseekocr2 import DeepseekOCR2ForCausalLM
    from transformers import AutoTokenizer

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 加载模型 - 使用bfloat16以匹配模型内部dtype转换
    model = DeepseekOCR2ForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,  # 模型配置指定bfloat16
        device_map=None,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )

    # 确保模型使用bfloat16并移动到CPU
    model = model.to(torch.bfloat16).cpu()
    model.eval()

    # 只执行一次的初始化操作（从 infer() 中移出）
    model.disable_torch_init()

    print(f"{Colors.GREEN}Model loaded successfully{Colors.RESET}")

    # 创建临时目录用于存储图像文件
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"{Colors.YELLOW}Processing {len(images)} pages...{Colors.RESET}")

        outputs_list = []
        draw_images = []

        total_start = time.time()
        for jdx, image in enumerate(tqdm(images, desc="Processing pages")):
            page_start = time.time()

            # 保存图像到临时文件
            temp_image_path = process_single_image(image, temp_dir)

            try:
                output_text = model.infer(
                    tokenizer=tokenizer,
                    prompt=PROMPT,
                    image_file=temp_image_path,
                    output_path=OUTPUT_PATH,
                    base_size=BASE_SIZE,
                    image_size=IMAGE_SIZE,
                    crop_mode=CROP_MODE,
                    eval_mode=True,
                    save_results=False,
                )

                outputs_list.append(output_text)

                # 处理检测结果并绘制边界框
                matches_ref, matches_images, mathes_other = re_match(output_text)
                result_image = process_image_with_refs(image.copy(), matches_ref, jdx)
                draw_images.append(result_image)

            except Exception as e:
                print(f"{Colors.RED}Error processing page {jdx}: {e}{Colors.RESET}")
                import traceback
                traceback.print_exc()
                outputs_list.append("")
                draw_images.append(image.copy())

            # 每页完成后清理内存
            gc.collect()

            page_time = time.time() - page_start
            elapsed = time.time() - total_start
            avg_per_page = elapsed / (jdx + 1)
            remaining = avg_per_page * (len(images) - jdx - 1)
            print(f"{Colors.BLUE}  Page {jdx}: {page_time:.1f}s | "
                  f"Avg: {avg_per_page:.1f}s/page | "
                  f"ETA: {remaining:.0f}s{Colors.RESET}")

        # 后处理：生成输出文件
        print(f"{Colors.YELLOW}Generating output files...{Colors.RESET}")

        output_path = OUTPUT_PATH
        mmd_det_path = (
            output_path + "/" + INPUT_PATH.split("/")[-1].replace(".pdf", "_det.mmd")
        )
        mmd_path = output_path + "/" + INPUT_PATH.split("/")[-1].replace("pdf", "mmd")
        pdf_out_path = (
            output_path
            + "/"
            + INPUT_PATH.split("/")[-1].replace(".pdf", "_layouts.pdf")
        )

        contents_det = ""
        contents = ""

        for jdx, (output, img) in enumerate(zip(outputs_list, images)):
            content = output

            if "" in content:  # repeat no eos
                content = content.replace("", "")
            else:
                if SKIP_REPEAT:
                    continue

            page_num = f"\n<--- Page Split --->"

            contents_det += content + f"\n{page_num}\n"

            matches_ref, matches_images, mathes_other = re_match(content)

            for idx, a_match_image in enumerate(matches_images):
                content = content.replace(
                    a_match_image,
                    f"![](images/" + str(jdx) + "_" + str(idx) + ".jpg)\n",
                )

            for idx, a_match_other in enumerate(mathes_other):
                content = (
                    content.replace(a_match_other, "")
                    .replace("\\coloneqq", ":=")
                    .replace("\\eqqcolon", "=:")
                    .replace("\n\n\n\n", "\n\n")
                    .replace("\n\n\n", "\n\n")
                )

            contents += content + f"\n{page_num}\n"

        with open(mmd_det_path, "w", encoding="utf-8") as afile:
            afile.write(contents_det)

        with open(mmd_path, "w", encoding="utf-8") as afile:
            afile.write(contents)

        pil_to_pdf_img2pdf(draw_images, pdf_out_path)

        print(f"{Colors.GREEN}Processing complete!{Colors.RESET}")
        print(f"Output files:")
        print(f"  - Detection results: {mmd_det_path}")
        print(f"  - Markdown results: {mmd_path}")
        print(f"  - Layout PDF: {pdf_out_path}")
        print(f"  - Extracted images: {OUTPUT_PATH}/images/")
