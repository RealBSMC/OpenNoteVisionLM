import os

BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6  # max:6
MAX_CONCURRENCY = (
    100  # If you have limited GPU memory, lower the concurrency count.
)
NUM_WORKERS = 64  # image pre-process (resize/padding) workers
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = os.environ.get("DEEPSEEK_OCR_MODEL_PATH", "./model_weights")  # Path to DeepSeek-OCR-2 model weights

# Batch processing scripts input paths:
# .pdf: run_dpsk_ocr_pdf.py;
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py;
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py


INPUT_PATH = os.environ.get("DEEPSEEK_OCR_INPUT_PATH", "./data/input.pdf")  # Default input path for batch scripts
OUTPUT_PATH = os.environ.get("DEEPSEEK_OCR_OUTPUT_PATH", "./data/output/")  # Default output directory for batch scripts

PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
# PROMPT = '<image>\nFree OCR.'
# .......


# tokenizer 在主脚本中按需加载，避免重复初始化
TOKENIZER = None
