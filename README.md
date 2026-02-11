# DeepSeek-OCR-2 Web App

> 中文文档请参阅 [README_CN.md](README_CN.md)

A modern web application that combines **DeepSeek-OCR-2** (state-of-the-art visual OCR) with **PageIndex** (vectorless, reasoning-based RAG) for intelligent PDF document processing and conversation.

## ✨ Features

### 📄 Advanced OCR
- **Visual Causal Flow OCR**: Uses DeepSeek-OCR-2 model for human-like document understanding
- **PDF to Markdown**: Converts PDF pages to structured Markdown with layout preservation
- **Image Extraction**: Automatically extracts and saves images from documents
- **Progress Tracking**: Real-time processing status with page-by-page updates

### 🤖 Intelligent Document Interaction
- **Reasoning-based RAG**: PageIndex-powered retrieval without vector databases or chunking
- **Tree Index Search**: Builds hierarchical document structure for human-like navigation
- **Multi-turn Conversation**: Ask questions about document content with context awareness
- **Semantic Search**: Find relevant sections using LLM reasoning rather than vector similarity

### 🌐 Modern Web Interface
- **FastAPI Backend**: High-performance async API server
- **Responsive Frontend**: Clean, intuitive interface for document management
- **Real-time Updates**: Live progress tracking and status monitoring
- **Multi-Provider Support**: Configure OpenAI, DeepSeek, OpenRouter, or custom LLM endpoints

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 8GB+ RAM (16GB recommended for OCR processing)
- DeepSeek-OCR-2 model weights (download separately)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/deepseek-ocr-2-app.git
cd deepseek-ocr-2-app
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download DeepSeek-OCR-2 model**
```bash
# Download from HuggingFace (requires git-lfs)
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR-2 ./model_weights
# Download from ModelScope （requires modelscope）
modelscope download --model deepseek-ai/DeepSeek-OCR-2 --local_dir ./model_weights
# Or download manually and extract to a directory
```

5. **Configure environment**
```bash
cp .env.example .env
# Edit .env to set your model path and API keys
```

### Configuration

Edit `.env` file:
```env
# DeepSeek-OCR-2 Model Path
DEEPSEEK_OCR_MODEL_PATH=/path/to/deepseek-ocr-2-weights

# RAG Configuration (choose one provider)
OPENAI_API_KEY=your_openai_api_key_here
# OR
DEEPSEEK_API_KEY=your_deepseek_api_key_here
# OR
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: RAG model selection
RAG_MODEL=gpt-4o-2024-11-20  # or deepseek-chat, anthropic/claude-3.5-sonnet, etc.
```

### Running the Application

```bash
# Start the web server (default port 8000)
python app.py

# Or with custom host/port
python app.py --host 0.0.0.0 --port 8000
```

Open your browser to: http://localhost:8000

## 📖 Usage Guide

### 1. Upload PDF
- Click "Upload PDF" button
- Select your PDF file
- System generates a unique document ID and starts processing

### 2. Monitor Processing
- View real-time progress with page completion status
- Watch as DeepSeek-OCR-2 converts each page to Markdown
- See extracted images in the gallery

### 3. Preview Results
- Browse through document pages
- View side-by-side: Original PDF vs OCR Markdown
- Check extracted images with captions

### 4. Build RAG Index
- Navigate to Settings tab
- Configure your LLM provider and API key
- Click "Build Index" to create document tree structure

### 5. Chat with Document
- Ask questions about document content
- Get answers with page references and reasoning
- Follow-up with related questions for multi-turn conversation

### 6. Search Document
- Search for specific content using semantic understanding
- Get ranked results with context snippets

## 🔧 Advanced Configuration

### Model Manager
The application uses lazy loading for DeepSeek-OCR-2 model:
- First PDF triggers model loading (takes 1-2 minutes)
- Subsequent documents reuse loaded model
- Models run on CPU by default (GPU optional)

### RAG Settings
Configure in web interface or via `config_manager.py`:
- **Provider**: OpenAI, DeepSeek, OpenRouter, or custom
- **Models**: GPT-4o, DeepSeek Chat, Claude, Gemini, etc.
- **Token Settings**: Context length and summary thresholds
- **Base URL**: Custom API endpoints for self-hosted models

### Performance Optimization
```python
# In config.py
BASE_SIZE = 1024        # Base image size for OCR
IMAGE_SIZE = 768        # Input size for model
MAX_CONCURRENCY = 100   # Adjust based on memory
NUM_WORKERS = 64        # Image processing workers
```

## 🏗️ Project Structure

```
deepseek-ocr-2-app/
├── app.py                 # FastAPI main application
├── config.py              # OCR parameters (image sizes, paths)
├── config_manager.py      # RAG configuration management
├── model_manager.py       # Singleton model loader (DeepSeek-OCR-2)
├── ocr_engine.py          # PDF→image→OCR pipeline
├── rag_engine.py          # PageIndex integration, indexing, Q&A
├── modeling_deepseekocr2.py # Custom model definition
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── LICENSE               # MIT License
└── README.md            # This file

data/                     # Auto-created at runtime
├── config/              # RAG configuration files
└── documents/           # Uploaded PDFs, metadata, indexes

static/                  # Web frontend
├── index.html          # Main interface
├── css/style.css       # Styling
└── js/app.js           # Frontend logic

DeepSeek-OCR-2/          # Original model code (submodule)
PageIndex/               # PageIndex RAG library (submodule)
deepencoderv2/           # Visual encoder components
process/                 # Preprocessing utilities
```

## 📊 Performance Notes

### OCR Processing Speed
- **First document**: ~1-2 minutes for model loading + OCR
- **Subsequent documents**: ~10-60 seconds per page (depends on content)
- **Memory usage**: 16GB+ RAM recommended
- **CPU optimization**: Automatically uses half of available CPU cores

### RAG Index Building
- **Index creation**: ~1-5 minutes (depends on document length and LLM)
- **Tree search**: ~2-10 seconds per query
- **Context handling**: Supports documents with 1000+ pages

## 🔒 Security Considerations

- **API Keys**: Stored encrypted in local config files
- **File Uploads**: Validated for PDF format only
- **Path Traversal**: Protected against directory traversal attacks
- **Data Privacy**: All processing happens locally; no document data sent externally (except for RAG API calls if configured)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest ruff black

# Run tests (to be implemented)
pytest

# Code formatting
black .
ruff check --fix .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- **DeepSeek-OCR-2**: Apache 2.0 License
- **PageIndex**: MIT License

## 🙏 Acknowledgments

- [DeepSeek AI](https://www.deepseek.com/) for the amazing OCR-2 model
- [Vectify AI](https://vectify.ai/) for PageIndex reasoning-based RAG
- All open-source libraries that make this project possible

## 🆘 Support

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Questions**: Check existing issues or start a discussion
- **Documentation**: Refer to this README and inline code comments

---

**Note**: This is a community project not officially affiliated with DeepSeek AI or Vectify AI. The DeepSeek-OCR-2 model weights must be downloaded separately from HuggingFace.