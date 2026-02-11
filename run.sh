#!/bin/bash
# DeepSeek-OCR-2 App启动脚本

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查Python版本
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" < "3.10" ]]; then
    echo "错误: 需要Python 3.10或更高版本，当前版本: $PYTHON_VERSION"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "未找到虚拟环境，正在创建..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "虚拟环境创建完成"
else
    source venv/bin/activate
fi

# 检查模型路径
if [ -z "$DEEPSEEK_OCR_MODEL_PATH" ]; then
    if [ -d "model_weights" ]; then
        export DEEPSEEK_OCR_MODEL_PATH="$PROJECT_ROOT/model_weights"
        echo "使用默认模型路径: $DEEPSEEK_OCR_MODEL_PATH"
    else
        echo "警告: 未设置DEEPSEEK_OCR_MODEL_PATH环境变量"
        echo "请下载DeepSeek-OCR-2模型权重并设置路径:"
        echo "  export DEEPSEEK_OCR_MODEL_PATH=/path/to/deepseek-ocr-2-weights"
        echo "或创建软链接:"
        echo "  ln -s /path/to/deepseek-ocr-2-weights ./model_weights"
        echo ""
        read -p "是否继续？(模型可能无法加载) [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# 检查RAG配置
if [ -f ".env" ]; then
    echo "加载.env文件"
    export $(grep -v '^#' .env | xargs)
else
    echo "警告: 未找到.env文件，使用默认配置"
    echo "创建.env文件: cp .env.example .env"
fi

# 启动应用
echo "启动DeepSeek-OCR-2 Web应用..."
echo "访问: http://localhost:8000"
echo "按Ctrl+C停止"
echo ""

python app.py