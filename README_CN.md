# OpenNoteVision LM / 开源视觉笔记 LM

> English documentation available at [README.md](README.md)

## 🎯 项目初衷：解决主流笔记AI的视觉短板

当你在使用 **NoteBookLM**、**opennotebook** 或其他文档对话工具时，是否遇到过这些烦恼？

❌ **扫描PDF无法识别** - 纯图像文档变成"哑巴"文件  
❌ **图文混合处理差** - 表格、图表、公式丢失信息  
❌ **布局理解有限** - 无法还原文档的视觉结构  
❌ **专业文档处理弱** - 学术论文、技术文档效果不佳

**OpenNoteVision LM** 正是为了解决这些痛点而生！我们结合了最先进的视觉 OCR 模型 **DeepSeek‑OCR‑2** 和推理式 RAG 框架 **PageIndex**，打造了一个真正能"看懂"扫描文档的智能笔记平台。

## 📊 与主流方案的对比

| 特性 | OpenNoteVision LM | NoteBookLM | opennotebook | 腾讯 iMA |
|------|-----------------|------------|--------------|----------|
| **扫描PDF处理** | ✅ 完美支持 | ❌ 不支持 | ❌ 不支持 | ⚠️ 有限支持 |
| **视觉OCR能力** | ✅ DeepSeek‑OCR‑2 | ❌ 无 | ❌ 无 | ⚠️ 基础OCR |
| **推理式检索** | ✅ PageIndex 树搜索 | ⚠️ 向量检索 | ⚠️ 向量检索 | ❓ 未知 |
| **开源程度** | ✅ 完全开源 | ❌ 闭源 | ✅ 开源 | ❌ 闭源 |
| **本地部署** | ✅ 支持 | ❌ 不支持 | ✅ 支持 | ❌ 不支持 |
| **多格式支持** | ✅ PDF/图像 | ⚠️ 有限 | ⚠️ 有限 | ✅ 多种格式 |
| **对话质量** | ✅ 上下文感知 | ✅ 良好 | ⚠️ 一般 | ❓ 未知 |

## ✨ 核心特色

### 👁️‍🗨️ **视觉优先的文档理解**
- **DeepSeek‑OCR‑2 模型**：业界领先的视觉因果流 OCR，理解文档如同人类
- **布局保留转换**：PDF → 结构化 Markdown，保持表格、图表、公式原貌
- **图像智能提取**：自动识别并保存文档中的图片、图表、示意图

### 🧠 **推理式智能对话**
- **PageIndex 树检索**：无需向量数据库，基于文档结构的推理式搜索
- **上下文感知问答**：理解文档整体结构，回答具有深度和关联性
- **多轮对话记忆**：保持对话历史，实现连贯的文档探索

### 🌐 **现代化全栈架构**
- **FastAPI 后端**：高性能异步 API，支持并发文档处理
- **响应式 Web 界面**：直观的文档管理、预览、对话界面
- **多 LLM 支持**：OpenAI、DeepSeek、OpenRouter、自定义端点

### 🔓 **开源与隐私**
- **完全开源**：代码透明，可审计，可自定义
- **本地优先**：所有数据处理在本地进行，保护隐私
- **自托管选项**：支持私有化部署，完全控制数据

## 🚀 快速开始

### 环境要求
- Python 3.10+
- 8GB+ 内存（推荐 16GB 以获得更好体验）
- DeepSeek‑OCR‑2 模型权重（需单独下载）

### 5分钟部署

1. **克隆项目**
```bash
git clone https://github.com/yourusername/OpenNoteVision-LM.git
cd OpenNoteVision-LM
```

2. **设置环境**
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

3. **下载模型**
```bash
# 下载 DeepSeek-OCR-2 模型权重
# 从 HuggingFace 下载（推荐）
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR-2 ./model_weights

# 或者手动下载后放置到 model_weights/ 目录
```

4. **配置应用**
```bash
# 复制配置文件模板
cp .env.example .env

# 编辑 .env 文件，至少设置模型路径
# DEEPSEEK_OCR_MODEL_PATH=./model_weights
```

5. **启动应用**
```bash
# 方式1：直接运行
python app.py

# 方式2：使用启动脚本（推荐）
./run.sh
```

6. **开始使用**
打开浏览器访问：http://localhost:8000

## 📖 使用场景

### 🎓 学术研究
- **扫描论文对话**：与扫描版学术论文进行深度问答
- **文献综述辅助**：快速提取多篇文献的核心观点
- **笔记自动整理**：将讲义、教材转换为结构化知识库

### 💼 商业办公
- **扫描合同分析**：快速理解合同条款和要点
- **报表数据提取**：从扫描的财务报表中提取关键数据
- **会议纪要处理**：将扫描的会议记录转换为可搜索文档

### 🏥 专业领域
- **医疗文档处理**：处理扫描的病历、检验报告
- **法律文件分析**：分析扫描的法律文书和判例
- **技术手册查询**：与扫描的技术文档进行交互式问答

### 👨‍💻 个人知识管理
- **读书笔记创建**：从扫描的书籍中提取精华内容
- **手写笔记数字化**：处理手写扫描笔记（需清晰）
- **个人档案管理**：建立可搜索的个人文档库

## 🏗️ 技术架构

```
OpenNoteVision-LM/
├── 视觉层 (Vision Layer)
│   ├── DeepSeek-OCR-2 ──── 视觉文档理解
│   ├── 图像预处理 ────── 优化扫描质量
│   └── 布局分析 ─────── 保留文档结构
│
├── 理解层 (Understanding Layer)  
│   ├── PageIndex ──────── 推理式文档索引
│   ├── 树状结构构建 ──── 文档语义组织
│   └── 上下文管理 ────── 对话状态维护
│
├── 交互层 (Interaction Layer)
│   ├── FastAPI 后端 ──── RESTful API 服务
│   ├── Web 前端 ──────── 用户界面
│   └── 多LLM适配器 ──── 支持多种大模型
│
└── 存储层 (Storage Layer)
    ├── 文档仓库 ──────── 原始文档存储
    ├── 索引数据库 ────── 树状索引持久化
    └── 对话历史 ──────── 用户交互记录
```

## 🔧 高级配置

### 性能优化
```python
# config.py 中的关键参数
BASE_SIZE = 1024        # 基础图像尺寸
IMAGE_SIZE = 768        # 模型输入尺寸
MAX_CONCURRENCY = 100   # 并发处理数（根据内存调整）
NUM_WORKERS = 64        # 图像处理线程数
```

### 多 LLM 配置
支持 OpenAI、DeepSeek、OpenRouter、自定义端点：
```env
# .env 文件配置示例
OPENAI_API_KEY=your_key_here
# 或
DEEPSEEK_API_KEY=your_key_here
# 或
OPENROUTER_API_KEY=your_key_here

# 模型选择
RAG_MODEL=gpt-4o-2024-11-20  # 或 deepseek-chat 等
```

### 自定义部署
- **Docker 部署**：提供 Dockerfile（待实现）
- **云服务部署**：支持 AWS、Azure、GCP
- **私有化部署**：企业内部网络部署

## 📈 性能表现

### 处理速度
- **模型首次加载**：1-2分钟（仅第一次）
- **扫描PDF处理**：10-30秒/页（取决于复杂程度）
- **索引构建**：1-5分钟（100页文档）
- **查询响应**：1-5秒（树检索优化）

### 资源占用
- **内存使用**：4-8GB（推荐 16GB）
- **CPU 使用**：自动优化，使用物理核心数
- **存储空间**：模型权重约 15GB，每文档额外 10-100MB

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 开发流程
1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m '添加了一些很棒的功能'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

### 开发环境
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码质量检查
ruff check .
black --check .
```

### 急需贡献的方向
- 📱 移动端适配
- 🐳 Docker 容器化
- 🌐 多语言界面
- 📊 性能基准测试
- 🔌 插件系统

## 📄 许可证

本项目采用 **MIT 许可证** - 详见 [LICENSE](LICENSE) 文件。

### 第三方组件许可证
- **DeepSeek‑OCR‑2**：Apache 2.0 许可证
- **PageIndex**：MIT 许可证
- **其他依赖**：各自的开源许可证

## 🙏 致谢

- **DeepSeek AI**：提供卓越的 DeepSeek‑OCR‑2 视觉模型
- **Vectify AI**：开发创新的 PageIndex 推理式 RAG 框架
- **开源社区**：所有使本项目成为可能的开源项目贡献者
- **早期用户**：提供宝贵反馈和改进建议

## 🆘 支持与反馈

- **问题报告**：[GitHub Issues](https://github.com/yourusername/OpenNoteVision-LM/issues)
- **功能建议**：通过 Issues 提交
- **技术讨论**：欢迎提交 Pull Request
- **使用问题**：查阅文档或提交 Issue

---

## 🚨 重要声明

**OpenNoteVision LM 是一个开源社区项目，并非以下产品的官方版本或衍生版本：**
- ❌ 不是 Google NoteBookLM 的开源替代
- ❌ 不是 opennotebook 的分支或改进版  
- ❌ 不是腾讯 iMA 的相关项目
- ❌ 与 DeepSeek AI、Vectify AI 无官方关联

**我们只是解决了这些产品未能满足的用户需求，提供了他们缺乏的视觉文档处理能力。**