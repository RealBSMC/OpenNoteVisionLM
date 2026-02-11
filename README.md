# OpenNoteVision LM / Open Source Visual Notebook LM

> 中文文档请参阅 [README_CN.md](README_CN.md)

## 🎯 Project Origin: Solving the Visual Limitations of Mainstream Notebook AIs

When using **NoteBookLM**, **opennotebook**, or other document conversation tools, have you encountered these frustrations?

❌ **Scanned PDFs cannot be recognized** - Pure image documents become "mute" files  
❌ **Poor mixed text-image processing** - Tables, charts, formulas lose information  
❌ **Limited layout understanding** - Cannot restore the visual structure of documents  
❌ **Weak professional document handling** - Poor performance with academic papers, technical documents

**OpenNoteVision LM** was born to solve these pain points! We combine the state-of-the-art visual OCR model **DeepSeek‑OCR‑2** with the reasoning-based RAG framework **PageIndex** to create an intelligent notebook platform that truly "understands" scanned documents.

## 📊 Comparison with Mainstream Solutions

| Feature | OpenNoteVision LM | NoteBookLM | opennotebook | Tencent iMA |
|---------|-----------------|------------|--------------|-------------|
| **Scanned PDF Processing** | ✅ Perfect support | ❌ Not supported | ❌ Not supported | ⚠️ Limited support |
| **Visual OCR Capability** | ✅ DeepSeek‑OCR‑2 | ❌ None | ❌ None | ⚠️ Basic OCR |
| **Reasoning-based Retrieval** | ✅ PageIndex tree search | ⚠️ Vector retrieval | ⚠️ Vector retrieval | ❓ Unknown |
| **Open Source** | ✅ Fully open source | ❌ Closed source | ✅ Open source | ❌ Closed source |
| **Local Deployment** | ✅ Supported | ❌ Not supported | ✅ Supported | ❌ Not supported |
| **Multi-format Support** | ✅ PDF/Images | ⚠️ Limited | ⚠️ Limited | ✅ Multiple formats |
| **Conversation Quality** | ✅ Context-aware | ✅ Good | ⚠️ Average | ❓ Unknown |

## ✨ Core Features

### 👁️‍🗨️ **Vision-First Document Understanding**
- **DeepSeek‑OCR‑2 Model**: Industry-leading visual causal flow OCR, understanding documents like humans
- **Layout-Preserving Conversion**: PDF → Structured Markdown, preserving tables, charts, formulas
- **Intelligent Image Extraction**: Automatically identify and save images, charts, diagrams from documents

### 🧠 **Reasoning-based Intelligent Conversation**
- **PageIndex Tree Retrieval**: No vector database needed, reasoning-based search based on document structure
- **Context-Aware Q&A**: Understand overall document structure, provide deep and relevant answers
- **Multi-turn Conversation Memory**: Maintain conversation history for coherent document exploration

### 🌐 **Modern Full-Stack Architecture**
- **FastAPI Backend**: High-performance async API supporting concurrent document processing
- **Responsive Web Interface**: Intuitive document management, preview, conversation interface
- **Multi-LLM Support**: OpenAI, DeepSeek, OpenRouter, custom endpoints

### 🔓 **Open Source & Privacy**
- **Fully Open Source**: Transparent code, auditable, customizable
- **Local-First**: All data processing happens locally, protecting privacy
- **Self-Hosting Options**: Support private deployment, full data control

## 🚀 Quick Start

### Requirements
- Python 3.10+
- 8GB+ RAM (16GB recommended for better experience)
- DeepSeek‑OCR‑2 model weights (need to download separately)

### 5-Minute Deployment

1. **Clone the Project**
```bash
git clone https://github.com/yourusername/OpenNoteVision-LM.git
cd OpenNoteVision-LM
```

2. **Setup Environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Download Models**
```bash
# Download DeepSeek-OCR-2 model weights
# From HuggingFace (recommended)
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR-2 ./model_weights

# Or manually download and place in model_weights/ directory
```

4. **Configure Application**
```bash
# Copy configuration template
cp .env.example .env

# Edit .env file, at least set model path
# DEEPSEEK_OCR_MODEL_PATH=./model_weights
```

5. **Start Application**
```bash
# Method 1: Direct run
python app.py

# Method 2: Using startup script (recommended)
./run.sh
```

6. **Start Using**
Open browser and visit: http://localhost:8000

## 📖 Use Cases

### 🎓 Academic Research
- **Scanned Paper Conversations**: Deep Q&A with scanned academic papers
- **Literature Review Assistance**: Quickly extract core viewpoints from multiple papers
- **Automatic Note Organization**: Convert lectures, textbooks into structured knowledge bases

### 💼 Business Office
- **Scanned Contract Analysis**: Quickly understand contract terms and key points
- **Report Data Extraction**: Extract key data from scanned financial reports
- **Meeting Minutes Processing**: Convert scanned meeting records into searchable documents

### 🏥 Professional Fields
- **Medical Document Processing**: Handle scanned medical records, test reports
- **Legal Document Analysis**: Analyze scanned legal documents and case law
- **Technical Manual Queries**: Interactive Q&A with scanned technical documents

### 👨‍💻 Personal Knowledge Management
- **Reading Note Creation**: Extract essence from scanned books
- **Handwritten Note Digitization**: Process scanned handwritten notes (needs to be clear)
- **Personal Archive Management**: Build searchable personal document libraries

## 🏗️ Technical Architecture

```
OpenNoteVision-LM/
├── Vision Layer
│   ├── DeepSeek-OCR-2 ──── Visual document understanding
│   ├── Image Preprocessing ─ Optimize scan quality
│   └── Layout Analysis ──── Preserve document structure
│
├── Understanding Layer  
│   ├── PageIndex ──────── Reasoning-based document indexing
│   ├── Tree Structure Building ─ Document semantic organization
│   └── Context Management ─ Conversation state maintenance
│
├── Interaction Layer
│   ├── FastAPI Backend ──── RESTful API service
│   ├── Web Frontend ─────── User interface
│   └── Multi-LLM Adapter ─ Support various large models
│
└── Storage Layer
    ├── Document Repository ─ Original document storage
    ├── Index Database ────── Tree index persistence
    └── Conversation History ─ User interaction records
```

## 🔧 Advanced Configuration

### Performance Optimization
```python
# Key parameters in config.py
BASE_SIZE = 1024        # Base image size
IMAGE_SIZE = 768        # Model input size
MAX_CONCURRENCY = 100   # Concurrent processing (adjust based on memory)
NUM_WORKERS = 64        # Image processing threads
```

### Multi-LLM Configuration
Support OpenAI, DeepSeek, OpenRouter, custom endpoints:
```env
# .env file configuration example
OPENAI_API_KEY=your_key_here
# or
DEEPSEEK_API_KEY=your_key_here
# or
OPENROUTER_API_KEY=your_key_here

# Model selection
RAG_MODEL=gpt-4o-2024-11-20  # or deepseek-chat, etc.
```

### Custom Deployment
- **Docker Deployment**: Provide Dockerfile (to be implemented)
- **Cloud Service Deployment**: Support AWS, Azure, GCP
- **Private Deployment**: Internal network deployment

## 📈 Performance

### Processing Speed
- **First Model Loading**: 1-2 minutes (only first time)
- **Scanned PDF Processing**: 10-30 seconds/page (depending on complexity)
- **Index Building**: 1-5 minutes (100-page document)
- **Query Response**: 1-5 seconds (tree retrieval optimized)

### Resource Usage
- **Memory Usage**: 4-8GB (16GB recommended)
- **CPU Usage**: Automatic optimization, uses physical cores
- **Storage Space**: Model weights ~15GB, additional 10-100MB per document

## 🤝 Contribution Guide

We welcome contributions in all forms!

### Development Process
1. Fork this repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add some amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Environment
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code quality check
ruff check .
black --check .
```

### Urgent Contribution Areas
- 📱 Mobile adaptation
- 🐳 Docker containerization
- 🌐 Multi-language interface
- 📊 Performance benchmarking
- 🔌 Plugin system

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Component Licenses
- **DeepSeek‑OCR‑2**: Apache 2.0 License
- **PageIndex**: MIT License
- **Other dependencies**: Respective open source licenses

## 🙏 Acknowledgments

- **DeepSeek AI**: For the excellent DeepSeek‑OCR‑2 visual model
- **Vectify AI**: For developing the innovative PageIndex reasoning-based RAG framework
- **Open Source Community**: All open source project contributors that made this project possible
- **Early Users**: For valuable feedback and improvement suggestions

## 🆘 Support & Feedback

- **Issue Reporting**: [GitHub Issues](https://github.com/yourusername/OpenNoteVision-LM/issues)
- **Feature Suggestions**: Submit via Issues
- **Technical Discussion**: Welcome to submit Pull Requests
- **Usage Problems**: Check documentation or submit Issue

---

## 🚨 Important Disclaimer

**OpenNoteVision LM is an open source community project and is NOT an official version or derivative of the following products:**
- ❌ NOT an open source alternative to Google NoteBookLM
- ❌ NOT a fork or improved version of opennotebook
- ❌ NOT related to Tencent iMA
- ❌ NO official affiliation with DeepSeek AI or Vectify AI

**We simply address user needs unmet by these products, providing visual document processing capabilities they lack.**