# DeepSeek-OCR-2 Web 应用

> English documentation available at [README.md](README.md)

一个现代化的 Web 应用程序，将 **DeepSeek‑OCR‑2**（先进的视觉 OCR）与 **PageIndex**（无向量、基于推理的 RAG）相结合，实现智能 PDF 文档处理和对话。

## ✨ 功能特色

### 📄 高级 OCR 功能
- **视觉因果流 OCR**：使用 DeepSeek‑OCR‑2 模型进行类人文档理解
- **PDF 转 Markdown**：将 PDF 页面转换为结构化 Markdown，保留布局
- **图像提取**：自动从文档中提取并保存图像
- **进度跟踪**：实时处理状态，逐页更新

### 🤖 智能文档交互
- **基于推理的 RAG**：PageIndex 驱动的检索，无需向量数据库或分块
- **树状索引搜索**：构建分层文档结构，实现类人导航
- **多轮对话**：基于上下文对文档内容提问
- **语义搜索**：使用 LLM 推理而非向量相似性查找相关章节

### 🌐 现代化 Web 界面
- **FastAPI 后端**：高性能异步 API 服务器
- **响应式前端**：简洁直观的文档管理界面
- **实时更新**：实时进度跟踪和状态监控
- **多提供商支持**：配置 OpenAI、DeepSeek、OpenRouter 或自定义 LLM 端点

## 🚀 快速开始

### 环境要求
- Python 3.10+
- 8GB+ 内存（OCR 处理推荐 16GB）
- DeepSeek‑OCR‑2 模型权重（需单独下载）

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/yourusername/deepseek-ocr-2-app.git
cd deepseek-ocr-2-app
```

2. **设置虚拟环境**
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **下载 DeepSeek‑OCR‑2 模型**
```bash
# 从 HuggingFace 下载（需要 git-lfs）
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR-2 ./model_weights

# 从 魔塔社区 下载 （需要modelscope）
modelscope download --model deepseek-ai/DeepSeek-OCR-2 --local_dir ./model_weights
# 或手动下载并解压到目录
```

5. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，设置模型路径和 API 密钥
```

### 配置说明

编辑 `.env` 文件：
```env
# DeepSeek-OCR-2 模型路径
DEEPSEEK_OCR_MODEL_PATH=/path/to/deepseek-ocr-2-weights

# RAG 配置（选择一个提供商）
OPENAI_API_KEY=your_openai_api_key_here
# 或
DEEPSEEK_API_KEY=your_deepseek_api_key_here
# 或
OPENROUTER_API_KEY=your_openrouter_api_key_here

# 可选：RAG 模型选择
RAG_MODEL=gpt-4o-2024-11-20  # 或 deepseek-chat, anthropic/claude-3.5-sonnet 等
```

### 运行应用

```bash
# 启动 Web 服务器（默认端口 8000）
python app.py

# 或使用自定义主机/端口
python app.py --host 0.0.0.0 --port 8000

# 或使用启动脚本
./run.sh
```

打开浏览器访问：http://localhost:8000

## 📖 使用指南

### 1. 上传 PDF
- 点击"上传 PDF"按钮
- 选择 PDF 文件
- 系统生成唯一文档 ID 并开始处理

### 2. 监控处理进度
- 查看实时进度和页面完成状态
- 观察 DeepSeek‑OCR‑2 将每页转换为 Markdown
- 在图库中查看提取的图像

### 3. 预览结果
- 浏览文档页面
- 并排查看：原始 PDF vs OCR Markdown
- 检查带标题的提取图像

### 4. 构建 RAG 索引
- 导航到设置标签页
- 配置 LLM 提供商和 API 密钥
- 点击"构建索引"创建文档树结构

### 5. 与文档对话
- 提问关于文档内容的问题
- 获取带页面引用和推理过程的答案
- 进行多轮相关问题的跟进对话

### 6. 搜索文档
- 使用语义理解搜索特定内容
- 获取带上下文片段的排序结果

## 🔧 高级配置

### 模型管理器
应用对 DeepSeek‑OCR‑2 模型使用懒加载：
- 第一个 PDF 触发模型加载（约 1-2 分钟）
- 后续文档重用已加载的模型
- 默认在 CPU 上运行（GPU 可选）

### RAG 设置
在 Web 界面或通过 `config_manager.py` 配置：
- **提供商**：OpenAI、DeepSeek、OpenRouter 或自定义
- **模型**：GPT‑4o、DeepSeek Chat、Claude、Gemini 等
- **Token 设置**：上下文长度和摘要阈值
- **基础 URL**：自托管模型的自定义 API 端点

### 性能优化
```python
# 在 config.py 中
BASE_SIZE = 1024        # OCR 基础图像尺寸
IMAGE_SIZE = 768        # 模型输入尺寸
MAX_CONCURRENCY = 100   # 根据内存调整
NUM_WORKERS = 64        # 图像处理工作线程数
```

## 🏗️ 项目结构

```
deepseek-ocr-2-app/
├── app.py                 # FastAPI 主应用
├── config.py              # OCR 参数（图像尺寸、路径）
├── config_manager.py      # RAG 配置管理
├── model_manager.py       # 单例模型加载器（DeepSeek‑OCR‑2）
├── ocr_engine.py          # PDF→图像→OCR 流水线
├── rag_engine.py          # PageIndex 集成、索引构建、问答
├── modeling_deepseekocr2.py # 自定义模型定义
├── requirements.txt       # Python 依赖
├── .env.example          # 环境变量模板
├── LICENSE               # MIT 许可证
├── README.md            # 英文文档
├── README_CN.md         # 中文文档
└── run.sh               # 一键启动脚本

data/                     # 运行时自动创建
├── config/              # RAG 配置文件
└── documents/           # 上传的 PDF、元数据、索引

static/                  # Web 前端
├── index.html          # 主界面
├── css/style.css       # 样式表
└── js/app.js           # 前端逻辑

DeepSeek-OCR-2/          # 原始模型代码（子模块）
PageIndex/               # PageIndex RAG 库（子模块）
deepencoderv2/           # 视觉编码器组件
process/                 # 预处理工具
tests/                   # 测试文件
```

## 📊 性能说明

### OCR 处理速度
- **首个文档**：模型加载 + OCR 约 1‑2 分钟
- **后续文档**：每页约 10‑60 秒（取决于内容）
- **内存使用**：推荐 16GB+ RAM
- **CPU 优化**：自动使用一半可用 CPU 核心

### RAG 索引构建
- **索引创建**：约 1‑5 分钟（取决于文档长度和 LLM）
- **树状搜索**：每个查询约 2‑10 秒
- **上下文处理**：支持 1000+ 页的文档

## 🔒 安全考虑

- **API 密钥**：存储在本地配置文件中加密
- **文件上传**：仅验证 PDF 格式
- **路径遍历**：防止目录遍历攻击
- **数据隐私**：所有处理均在本地进行；文档数据不会外部发送（除非配置了 RAG API 调用）

## 🤝 贡献指南

欢迎贡献！请随时提交 Pull Request。

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m '添加了一些很棒的功能'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码格式化
black .
ruff check --fix .
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

### 第三方许可证
- **DeepSeek‑OCR‑2**：Apache 2.0 许可证
- **PageIndex**：MIT 许可证

## 🙏 致谢

- [DeepSeek AI](https://www.deepseek.com/) 提供卓越的 OCR‑2 模型
- [Vectify AI](https://vectify.ai/) 提供 PageIndex 推理式 RAG
- 所有使本项目成为可能的开源库

## 🆘 支持

- **问题报告**：使用 GitHub Issues 报告错误和功能请求
- **问题咨询**：查看现有问题或开始讨论
- **文档参考**：参考本 README 和代码内注释

---

**注意**：这是一个社区项目，并非 DeepSeek AI 或 Vectify AI 的官方项目。DeepSeek‑OCR‑2 模型权重需从 HuggingFace 单独下载。
