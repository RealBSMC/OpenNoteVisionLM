/* ============================================================
   DeepSeek OCR 2 - Frontend Application
   ============================================================ */

// ============================================================
// Markdown + KaTeX Renderer
// ============================================================

const md = window.markdownit({
    html: true,
    linkify: true,
    typographer: true,
    breaks: true,
});

function renderMathMarkdown(text, docId) {
    if (!text) return '<p class="text-muted">暂无内容</p>';

    const mathBlocks = [];
    const mathInlines = [];

    // ---- 块级公式提取（优先级从高到低） ----

    // 1a. \[...\] 块级公式（OCR 模型常用格式）
    text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => {
        mathBlocks.push(math.trim());
        return `\n%%MBLK_${mathBlocks.length - 1}%%\n`;
    });

    // 1b. $$...$$ 块级公式
    text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
        mathBlocks.push(math.trim());
        return `\n%%MBLK_${mathBlocks.length - 1}%%\n`;
    });

    // 1c. \begin{...}...\end{...} 独立环境（equation, align, gather 等）
    text = text.replace(/\\begin\{(equation|align|aligned|gather|gathered|multline|split|cases|matrix|pmatrix|bmatrix|vmatrix|array)\*?\}([\s\S]*?)\\end\{\1\*?\}/g, (match, env, math) => {
        // 保留完整环境给 KaTeX
        mathBlocks.push(match.trim());
        return `\n%%MBLK_${mathBlocks.length - 1}%%\n`;
    });

    // ---- 行内公式提取 ----

    // 2a. \(...\) 行内公式（OCR 模型常用格式）
    text = text.replace(/\\\(([\s\S]*?)\\\)/g, (_, math) => {
        mathInlines.push(math.trim());
        return `%%MINL_${mathInlines.length - 1}%%`;
    });

    // 2b. $...$ 行内公式（不匹配 $$）
    text = text.replace(/(?<!\$)\$(?!\$)((?:[^$\\]|\\.)+?)\$(?!\$)/g, (_, math) => {
        mathInlines.push(math.trim());
        return `%%MINL_${mathInlines.length - 1}%%`;
    });

    // 3. 转换图片路径：images/X_Y.jpg → API 路径
    if (docId) {
        text = text.replace(
            /!\[([^\]]*)\]\(images\/([^)]+)\)/g,
            `![$1](/api/documents/${docId}/images/$2)`
        );
    }

    // 4. 渲染 Markdown
    let html = md.render(text);

    // 5. 替换块级公式占位符
    mathBlocks.forEach((math, i) => {
        try {
            const rendered = katex.renderToString(math, {
                displayMode: true,
                throwOnError: false,
                trust: true,
                macros: {
                    "\\coloneqq": ":=",
                    "\\eqqcolon": "=:",
                },
            });
            html = html.replace(`%%MBLK_${i}%%`, rendered);
        } catch (e) {
            html = html.replace(`%%MBLK_${i}%%`, `<pre class="math-error" title="${e.message}">$$${math}$$</pre>`);
        }
    });

    // 6. 替换行内公式占位符
    mathInlines.forEach((math, i) => {
        try {
            const rendered = katex.renderToString(math, {
                displayMode: false,
                throwOnError: false,
                trust: true,
                macros: {
                    "\\coloneqq": ":=",
                    "\\eqqcolon": "=:",
                },
            });
            html = html.replace(`%%MINL_${i}%%`, rendered);
        } catch (e) {
            html = html.replace(`%%MINL_${i}%%`, `<code class="math-error" title="${e.message}">${math}</code>`);
        }
    });

    // 清理 <p> 包裹的 katex-display
    html = html.replace(/<p>([\s\S]*?class="katex-display"[\s\S]*?)<\/p>/g, '$1');

    return html;
}


// ============================================================
// API Helper
// ============================================================

async function api(path, options = {}) {
    const resp = await fetch(path, options);
    if (!resp.ok) {
        const err = await resp.text();
        throw new Error(`API error ${resp.status}: ${err}`);
    }
    return resp.json();
}


// ============================================================
// State
// ============================================================

const state = {
    currentView: 'library',   // 'library' | 'viewer'
    documents: [],
    pollTimer: null,
    // Viewer state
    viewerDocId: null,
    viewerDoc: null,
    currentPage: 0,
    pageCache: {},
    // RAG state
    ragConfigured: false,
    chatVisible: false,
    chatHistory: [],
    indexStatus: 'not_built',
    providersData: null,
};


// ============================================================
// DOM References
// ============================================================

const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    libraryView: $('#library-view'),
    viewerView: $('#viewer-view'),
    documentList: $('#document-list'),
    emptyState: $('#empty-state'),
    uploadBtn: $('#upload-btn'),
    uploadModal: $('#upload-modal'),
    modalClose: $('#modal-close'),
    dropZone: $('#drop-zone'),
    fileInput: $('#file-input'),
    uploadProgress: $('#upload-progress'),
    uploadFileName: $('#upload-file-name'),
    uploadProgressBar: $('#upload-progress-bar'),
    uploadStatus: $('#upload-status'),
    modelBadge: $('#model-status-badge'),
    ragBadge: $('#rag-status-badge'),
    settingsBtn: $('#settings-btn'),
    settingsModal: $('#settings-modal'),
    settingsClose: $('#settings-close'),
    providerSelect: $('#provider-select'),
    providerHint: $('#provider-hint'),
    customUrlGroup: $('#custom-url-group'),
    customUrlInput: $('#custom-url-input'),
    apiKeyInput: $('#api-key-input'),
    modelPathInput: $('#model-path-input'),
    browseModelPathBtn: $('#browse-model-path-btn'),
    apiKeyHint: $('#api-key-hint'),
    indexModelInput: $('#index-model-input'),
    indexModelHint: $('#index-model-hint'),
    customIndexModelGroup: $('#custom-index-model-group'),
    customIndexModelInput: $('#custom-index-model-input'),
    chatModelInput: $('#chat-model-input'),
    chatModelHint: $('#chat-model-hint'),
    customChatModelGroup: $('#custom-chat-model-group'),
    customChatModelInput: $('#custom-chat-model-input'),
    summaryThresholdInput: $('#summary-threshold-input'),
    contextLengthInput: $('#context-length-input'),
    saveSettingsBtn: $('#save-settings-btn'),
    settingsStatus: $('#settings-status'),
    // Browse modal
    browseModal: $('#browse-modal'),
    browseClose: $('#browse-close'),
    browseCurrentPath: $('#browse-current-path'),
    browseEntries: $('#browse-entries'),
    browseParentBtn: $('#browse-parent-btn'),
    browseCancelBtn: $('#browse-cancel-btn'),
    browseSelectBtn: $('#browse-select-btn'),
    // Viewer
    backBtn: $('#back-btn'),
    docTitle: $('#doc-title'),
    prevPage: $('#prev-page'),
    nextPage: $('#next-page'),
    pageInput: $('#page-input'),
    pageTotal: $('#page-total'),
    viewerStatus: $('#viewer-status'),
    markdownBody: $('#markdown-body'),
    pdfPageImage: $('#pdf-page-image'),
    resizeHandle: $('#resize-handle'),
    markdownPane: $('#markdown-pane'),
    pdfPane: $('#pdf-pane'),
    viewerContent: $('#viewer-content'),
    // Chat
    buildIndexBtn: $('#build-index-btn'),
    toggleChatBtn: $('#toggle-chat-btn'),
    chatPane: $('#chat-pane'),
    chatMessages: $('#chat-messages'),
    chatInput: $('#chat-input'),
    chatSendBtn: $('#chat-send-btn'),
    indexStatusBadge: $('#index-status-badge'),
};

// Provider 配置信息
const PROVIDER_INFO = {
    openai: {
        hint: 'OpenAI 官方 API',
        keyHint: '从 platform.openai.com 获取 API Key',
        keyPlaceholder: 'sk-...',
    },
    openrouter: {
        hint: '聚合多个 LLM 提供商，支持 GPT、Claude、Gemini 等',
        keyHint: '从 openrouter.ai 获取 API Key',
        keyPlaceholder: 'sk-or-...',
    },
    deepseek: {
        hint: 'DeepSeek 官方 API，性价比高',
        keyHint: '从 platform.deepseek.com 获取 API Key',
        keyPlaceholder: 'sk-...',
    },
    custom: {
        hint: '使用自定义 API 端点（兼容 OpenAI 格式）',
        keyHint: '输入您的 API Key',
        keyPlaceholder: 'API Key',
    },
};


// ============================================================
// View Switching
// ============================================================

function showView(name) {
    state.currentView = name;
    dom.libraryView.classList.toggle('active', name === 'library');
    dom.viewerView.classList.toggle('active', name === 'viewer');

    if (name === 'library') {
        startLibraryPolling();
    } else {
        stopLibraryPolling();
    }
}


// ============================================================
// Library View
// ============================================================

async function loadDocuments() {
    try {
        state.documents = await api('/api/documents');
        renderDocumentList();
    } catch (e) {
        console.error('Failed to load documents:', e);
    }
}

function renderDocumentList() {
    const docs = state.documents;
    dom.emptyState.style.display = docs.length === 0 ? 'flex' : 'none';
    dom.documentList.style.display = docs.length === 0 ? 'none' : 'grid';

    dom.documentList.innerHTML = docs.map(doc => {
        const statusInfo = getStatusDisplay(doc);
        const timeStr = formatTime(doc.created_at);
        const progress = doc.page_count > 0 ? Math.round((doc.processed_pages / doc.page_count) * 100) : 0;

        return `
        <div class="doc-card" data-id="${doc.id}">
            <div class="doc-card-header">
                <div class="doc-card-icon">
                    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5">
                        <rect x="4" y="2" width="12" height="16" rx="2"/>
                        <path d="M7 6h6M7 9h6M7 12h4"/>
                    </svg>
                </div>
                <button class="btn btn-danger delete-btn" data-id="${doc.id}" title="删除">
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M4 4l8 8M12 4l-8 8"/>
                    </svg>
                </button>
            </div>
            <div class="doc-card-title" title="${doc.filename}">${doc.filename}</div>
            <div class="doc-card-meta">
                <span>${doc.page_count} 页</span>
                <span>${timeStr}</span>
                ${doc.processing_time ? `<span>${doc.processing_time}s</span>` : ''}
            </div>
            <div class="doc-card-footer">
                <span class="badge ${statusInfo.badgeClass}">${statusInfo.label}</span>
            </div>
            ${(doc.status === 'processing' || doc.status === 'loading_model') ? `
            <div class="doc-card-progress">
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width:${doc.status === 'loading_model' ? 0 : progress}%"></div>
                </div>
                <div class="progress-text">${doc.status === 'loading_model' ? '正在加载模型...' : `${doc.processed_pages}/${doc.page_count} 页 (${progress}%)`}</div>
            </div>
            ` : ''}
        </div>`;
    }).join('');

    // 绑定卡片点击
    dom.documentList.querySelectorAll('.doc-card').forEach(card => {
        card.addEventListener('click', (e) => {
            if (e.target.closest('.delete-btn')) return;
            openViewer(card.dataset.id);
        });
    });

    // 绑定删除按钮
    dom.documentList.querySelectorAll('.delete-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const id = btn.dataset.id;
            if (confirm('确定删除此文档及其所有结果？')) {
                try {
                    await api(`/api/documents/${id}`, { method: 'DELETE' });
                    loadDocuments();
                } catch (err) {
                    alert('删除失败: ' + err.message);
                }
            }
        });
    });
}

function getStatusDisplay(doc) {
    switch (doc.status) {
        case 'queued':       return { label: '排队中', badgeClass: 'badge-gray' };
        case 'loading_model': return { label: '加载模型', badgeClass: 'badge-yellow' };
        case 'processing':   return { label: '处理中', badgeClass: 'badge-blue' };
        case 'completed':    return { label: '已完成', badgeClass: 'badge-green' };
        case 'error':        return { label: '失败', badgeClass: 'badge-red' };
        default:             return { label: doc.status, badgeClass: 'badge-gray' };
    }
}

function formatTime(isoStr) {
    if (!isoStr) return '';
    const d = new Date(isoStr);
    const now = new Date();
    const diff = (now - d) / 1000;
    if (diff < 60) return '刚刚';
    if (diff < 3600) return `${Math.floor(diff/60)} 分钟前`;
    if (diff < 86400) return `${Math.floor(diff/3600)} 小时前`;
    return d.toLocaleDateString('zh-CN');
}

// 轮询：检查正在处理的文档的进度
let pollInterval = null;

function startLibraryPolling() {
    stopLibraryPolling();
    pollInterval = setInterval(() => {
        const hasActive = state.documents.some(d =>
            ['queued', 'loading_model', 'processing'].includes(d.status)
        );
        if (hasActive) {
            loadDocuments();
        }
    }, 2000);
}

function stopLibraryPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// 模型状态
async function updateModelStatus() {
    try {
        const data = await api('/api/model/status');
        const badge = dom.modelBadge;
        switch (data.status) {
            case 'loaded':
                badge.className = 'badge badge-green';
                badge.textContent = '模型已加载';
                break;
            case 'loading':
                badge.className = 'badge badge-yellow';
                badge.textContent = '模型加载中...';
                break;
            case 'error':
                badge.className = 'badge badge-red';
                badge.textContent = '模型加载失败';
                break;
            default:
                badge.className = 'badge badge-gray';
                badge.textContent = '模型未加载';
        }
    } catch {}
}

// RAG 配置状态
async function updateRagStatus() {
    try {
        const data = await api('/api/settings');
        state.ragConfigured = data.is_configured;
        const badge = dom.ragBadge;
        
        if (data.is_configured) {
            badge.className = 'badge badge-green';
            badge.textContent = 'RAG 已配置';
        } else {
            badge.className = 'badge badge-gray';
            badge.textContent = 'RAG 未配置';
        }
        
    // 更新设置表单 - 注意：后端返回的是 index_model 和 chat_model
    // 这个函数只更新状态，实际表单在 loadSettings 中处理
    } catch {}
}

// 设置模态框
function showSettingsModal() {
    dom.settingsModal.style.display = 'flex';
    dom.settingsStatus.className = 'settings-status';
    dom.settingsStatus.textContent = '';
    loadSettings();
}

function hideSettingsModal() {
    dom.settingsModal.style.display = 'none';
}

async function loadSettings() {
    try {
        const [settings, providers] = await Promise.all([
            api('/api/settings'),
            api('/api/providers'),
        ]);
        
        state.providersData = providers;
        
    // 先设置provider，然后更新模型列表
    const provider = settings.provider || 'openai';
    dom.providerSelect.value = provider;
    onProviderChange(provider);
    
    // 使用微任务在DOM更新后设置模型值
    setTimeout(() => {
        if (settings.index_model) {
            if (provider === 'custom') {
                dom.customIndexModelInput.value = settings.index_model;
            } else {
                const indexOption = dom.indexModelInput.querySelector(`option[value="${settings.index_model}"]`);
                if (indexOption) {
                    dom.indexModelInput.value = settings.index_model;
                } else if (dom.indexModelInput.options.length > 1) {
                    // 如果没有完全匹配的选项，但选择框有选项，选择第一个非默认选项
                    dom.indexModelInput.value = dom.indexModelInput.options[1].value;
                }
            }
        }
        if (settings.chat_model) {
            if (provider === 'custom') {
                dom.customChatModelInput.value = settings.chat_model;
            } else {
                const chatOption = dom.chatModelInput.querySelector(`option[value="${settings.chat_model}"]`);
                if (chatOption) {
                    dom.chatModelInput.value = settings.chat_model;
                } else if (dom.chatModelInput.options.length > 1) {
                    // 如果没有完全匹配的选项，但选择框有选项，选择第一个非默认选项
                    dom.chatModelInput.value = dom.chatModelInput.options[1].value;
                }
            }
        }
    }, 0);
        
        if (settings.api_base_url) {
            dom.customUrlInput.value = settings.api_base_url;
        }
        
        if (settings.model_path) {
            dom.modelPathInput.value = settings.model_path;
        }
        
        if (settings.context_length) {
            dom.contextLengthInput.value = settings.context_length;
        }
        if (settings.summary_token_threshold) {
            dom.summaryThresholdInput.value = settings.summary_token_threshold;
        }
    } catch (e) {
        console.error('Failed to load settings:', e);
    }
}

function onProviderChange(provider) {
    const info = PROVIDER_INFO[provider] || PROVIDER_INFO.openai;
    
    dom.providerHint.textContent = info.hint;
    dom.apiKeyHint.textContent = info.keyHint;
    dom.apiKeyInput.placeholder = info.keyPlaceholder;
    
    // 显示/隐藏自定义 URL
    dom.customUrlGroup.style.display = provider === 'custom' ? 'flex' : 'none';
    
    // 更新模型列表
    if (state.providersData) {
        updateModelSelect(provider);
    }
}

function updateModelSelect(provider) {
    const models = state.providersData?.models_by_provider?.[provider] || [];
    
    // 清空并重新填充索引模型选择框
    dom.indexModelInput.innerHTML = '';
    // 保留HTML中的默认选项
    const indexDefaultOption = document.createElement('option');
    indexDefaultOption.value = '';
    indexDefaultOption.textContent = '请选择模型';
    dom.indexModelInput.appendChild(indexDefaultOption);
    
    // 清空并重新填充聊天模型选择框
    dom.chatModelInput.innerHTML = '';
    const chatDefaultOption = document.createElement('option');
    chatDefaultOption.value = '';
    chatDefaultOption.textContent = '请选择模型';
    dom.chatModelInput.appendChild(chatDefaultOption);
    
    if (provider === 'custom') {
        // 自定义模式：显示自定义输入框，隐藏选择框
        dom.customIndexModelGroup.style.display = 'flex';
        dom.customChatModelGroup.style.display = 'flex';
        dom.indexModelInput.style.display = 'none';
        dom.chatModelInput.style.display = 'none';
        dom.indexModelHint.textContent = '输入索引模型标识符';
        dom.chatModelHint.textContent = '输入问答模型标识符';
    } else {
        // 标准模式：显示选择框，隐藏自定义输入框
        dom.customIndexModelGroup.style.display = 'none';
        dom.customChatModelGroup.style.display = 'none';
        dom.indexModelInput.style.display = 'block';
        dom.chatModelInput.style.display = 'block';
        dom.indexModelHint.textContent = '用于构建文档索引（需要较强能力）';
        dom.chatModelHint.textContent = '用于回答问题（可用较快模型）';
        
        // 填充两个选择框
        models.forEach(model => {
            // 索引模型选项
            const indexOption = document.createElement('option');
            indexOption.value = model.value;
            indexOption.textContent = model.label;
            indexOption.title = model.description || '';
            dom.indexModelInput.appendChild(indexOption.cloneNode(true));
            
            // 聊天模型选项
            const chatOption = indexOption.cloneNode(true);
            dom.chatModelInput.appendChild(chatOption);
        });
    }
}

async function saveSettings() {
    const provider = dom.providerSelect.value;
    const apiKey = dom.apiKeyInput.value.trim();
    const indexModel = provider === 'custom'
        ? dom.customIndexModelInput.value.trim()
        : dom.indexModelInput.value;
    const chatModel = provider === 'custom'
        ? dom.customChatModelInput.value.trim()
        : dom.chatModelInput.value;
    const threshold = parseInt(dom.summaryThresholdInput.value, 10);
    const contextLength = parseInt(dom.contextLengthInput.value, 10);
    const customUrl = dom.customUrlInput.value.trim();
    const modelPath = dom.modelPathInput.value.trim();

    if (provider === 'custom' && !customUrl) {
        dom.settingsStatus.className = 'settings-status error';
        dom.settingsStatus.textContent = '自定义模式需要填写 API Base URL';
        return;
    }

    if (!indexModel) {
        dom.settingsStatus.className = 'settings-status error';
        dom.settingsStatus.textContent = '请选择或输入索引模型';
        return;
    }

    if (!chatModel) {
        dom.settingsStatus.className = 'settings-status error';
        dom.settingsStatus.textContent = '请选择或输入问答模型';
        return;
    }

    dom.saveSettingsBtn.disabled = true;
    dom.saveSettingsBtn.textContent = '保存中...';

    try {
         const body = {
             provider: provider,
             index_model: indexModel,
             chat_model: chatModel,
             model_path: modelPath,
             summary_token_threshold: threshold,
             context_length: contextLength || 8192,
         };

        if (provider === 'custom' && customUrl) {
            body.api_base_url = customUrl;
        }

        if (apiKey) {
            body.api_key = apiKey;
        }

        const resp = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(err);
        }

        const data = await resp.json();

        dom.settingsStatus.className = 'settings-status success';
        dom.settingsStatus.textContent = '设置保存成功！';
        dom.apiKeyInput.value = '';

        updateRagStatus();

        setTimeout(hideSettingsModal, 1000);
    } catch (e) {
        dom.settingsStatus.className = 'settings-status error';
        dom.settingsStatus.textContent = '保存失败: ' + e.message;
    } finally {
        dom.saveSettingsBtn.disabled = false;
        dom.saveSettingsBtn.textContent = '保存设置';
    }
}


// ============================================================
// Directory Browser
// ============================================================

let browseCurrentPath = '';

async function browseDirectory() {
    try {
        const resp = await fetch('/api/browse-directory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ current_path: dom.modelPathInput.value || '' }),
        });

        if (!resp.ok) {
            throw new Error('Failed to browse directory');
        }

        const data = await resp.json();

        if (data.method === 'dialog') {
            // 系统对话框成功选择了路径
            dom.modelPathInput.value = data.selected_path;
        } else if (data.method === 'browse') {
            // 系统对话框不可用，显示 Web 目录浏览器
            showBrowseModal(data);
        } else if (data.method === 'error') {
            alert(data.message);
        }
    } catch (e) {
        console.error('Browse directory failed:', e);
        alert('浏览目录失败: ' + e.message);
    }
}

function showBrowseModal(data) {
    browseCurrentPath = data.current_path;
    dom.browseCurrentPath.value = browseCurrentPath;
    dom.browseParentBtn.disabled = !data.parent_path;
    renderBrowseEntries(data.entries);
    dom.browseModal.style.display = 'flex';
}

function hideBrowseModal() {
    dom.browseModal.style.display = 'none';
}

function renderBrowseEntries(entries) {
    if (entries.length === 0) {
        dom.browseEntries.innerHTML = '<div class="browse-empty">此目录没有子目录</div>';
        return;
    }

    dom.browseEntries.innerHTML = entries.map(entry => `
        <div class="browse-entry" data-path="${entry.path}">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M2 4h12v10H2z"/>
                <path d="M2 6h12"/>
                <path d="M6 4V2h4v2"/>
            </svg>
            <span>${entry.name}</span>
        </div>
    `).join('');

    // 绑定点击事件
    dom.browseEntries.querySelectorAll('.browse-entry').forEach(el => {
        el.addEventListener('dblclick', async () => {
            await navigateToDirectory(el.dataset.path);
        });
        el.addEventListener('click', () => {
            dom.browseEntries.querySelectorAll('.browse-entry').forEach(e => e.classList.remove('selected'));
            el.classList.add('selected');
        });
    });
}

async function navigateToDirectory(path) {
    try {
        const resp = await fetch('/api/browse-directory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ current_path: path }),
        });

        if (!resp.ok) {
            throw new Error('Failed to navigate');
        }

        const data = await resp.json();

        if (data.method === 'browse') {
            browseCurrentPath = data.current_path;
            dom.browseCurrentPath.value = browseCurrentPath;
            dom.browseParentBtn.disabled = !data.parent_path;
            renderBrowseEntries(data.entries);
        } else if (data.method === 'error') {
            alert(data.message);
        }
    } catch (e) {
        console.error('Navigate failed:', e);
        alert('无法访问目录: ' + e.message);
    }
}

async function navigateToParent() {
    if (!browseCurrentPath || browseCurrentPath === '/') return;
    const parentPath = browseCurrentPath.split('/').slice(0, -1).join('/') || '/';
    await navigateToDirectory(parentPath);
}

function selectCurrentDirectory() {
    dom.modelPathInput.value = browseCurrentPath;
    hideBrowseModal();
}


// ============================================================
// RAG / Chat
// ============================================================

async function checkIndexStatus() {
    if (!state.viewerDocId) return;
    
    try {
        const data = await api(`/api/documents/${state.viewerDocId}/index-status`);
        state.indexStatus = data.index_status;
        updateIndexBadge();
        
        if (state.indexStatus === 'completed') {
            dom.chatInput.disabled = false;
            dom.chatSendBtn.disabled = false;
        } else {
            dom.chatInput.disabled = true;
            dom.chatSendBtn.disabled = true;
        }
    } catch {}
}

function updateIndexBadge() {
    const badge = dom.indexStatusBadge;
    switch (state.indexStatus) {
        case 'building':
            badge.className = 'badge badge-yellow badge-sm';
            badge.innerHTML = '<span class="spinner-sm"></span> 构建中...';
            break;
        case 'completed':
            badge.className = 'badge badge-green badge-sm';
            badge.textContent = '已建索引';
            break;
        case 'error':
            badge.className = 'badge badge-red badge-sm';
            badge.textContent = '构建失败';
            break;
        default:
            badge.className = 'badge badge-gray badge-sm';
            badge.textContent = '未建索引';
    }
}

async function buildIndex() {
    if (!state.viewerDocId) return;
    
    if (!state.ragConfigured) {
        alert('请先在设置中配置 API Key');
        showSettingsModal();
        return;
    }
    
    dom.buildIndexBtn.disabled = true;
    dom.buildIndexBtn.innerHTML = '<span class="spinner-sm"></span> 构建中...';
    
    const buildingMsg = document.createElement('div');
    buildingMsg.className = 'chat-message system building-msg';
    buildingMsg.innerHTML = '<span class="spinner-sm"></span> 正在构建索引，这可能需要几分钟，请耐心等待...';
    dom.chatMessages.appendChild(buildingMsg);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    
    if (!state.chatVisible) {
        state.chatVisible = true;
        dom.chatPane.style.display = 'flex';
    }
    
    try {
        const resp = await api(`/api/documents/${state.viewerDocId}/build-index`, { method: 'POST' });
        
        if (resp.status === 'already_exists') {
            state.indexStatus = 'completed';
            updateIndexBadge();
            buildingMsg.innerHTML = '✓ 索引已存在，无需重新构建';
        } else {
            state.indexStatus = 'building';
            updateIndexBadge();
            pollIndexStatus(buildingMsg);
        }
    } catch (e) {
        buildingMsg.className = 'chat-message error';
        buildingMsg.textContent = '构建索引失败: ' + e.message;
        alert('构建索引失败: ' + e.message);
    } finally {
        dom.buildIndexBtn.disabled = false;
        dom.buildIndexBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M2 4h12v10H2z"/>
                <path d="M2 7h12M5 2v2M11 2v2"/>
            </svg>
            构建索引
        `;
    }
}

let indexPollInterval = null;

function stopIndexPolling() {
    if (indexPollInterval) {
        clearInterval(indexPollInterval);
        indexPollInterval = null;
    }
}

function pollIndexStatus(buildingMsg) {
    stopIndexPolling();  // 先清理旧的轮询
    indexPollInterval = setInterval(async () => {
        await checkIndexStatus();
        if (state.indexStatus === 'completed') {
            stopIndexPolling();
            if (buildingMsg) {
                buildingMsg.className = 'chat-message system';
                buildingMsg.innerHTML = '✓ 索引构建完成！现在可以开始问答了。';
            }
            dom.chatInput.disabled = false;
            dom.chatSendBtn.disabled = false;
        } else if (state.indexStatus === 'error') {
            stopIndexPolling();
            if (buildingMsg) {
                buildingMsg.className = 'chat-message error';
                buildingMsg.textContent = '索引构建失败，请查看日志或重试';
            }
        } else {
            if (buildingMsg) {
                buildingMsg.innerHTML = '<span class="spinner-sm"></span> 正在构建索引，请耐心等待...';
            }
        }
    }, 2000);
}

function toggleChat() {
    state.chatVisible = !state.chatVisible;
    dom.chatPane.style.display = state.chatVisible ? 'flex' : 'none';
    
    if (state.chatVisible && state.chatHistory.length === 0) {
        addChatMessage('system', '请先构建索引，然后输入问题进行问答。');
    }
}

function addChatMessage(role, content, extra = {}) {
    state.chatHistory.push({ role, content, ...extra });
    
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message ${role}`;
    
    if (role === 'assistant') {
        let html = renderMathMarkdown(content, state.viewerDocId);
        if (extra.usage) {
            const u = extra.usage;
            html += `<div class="chat-usage">Token 消耗: ${u.total_tokens} (输入 ${u.prompt_tokens} / 输出 ${u.completion_tokens})</div>`;
        }
        msgDiv.innerHTML = html;
    } else if (role === 'loading') {
        msgDiv.innerHTML = `<span class="spinner-sm"></span> ${content}`;
    } else {
        msgDiv.textContent = content;
    }
    
    dom.chatMessages.appendChild(msgDiv);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    return msgDiv;
}

function updateChatMessage(msgDiv, role, content, extra = {}) {
    msgDiv.className = `chat-message ${role}`;
    if (role === 'assistant') {
        let html = renderMathMarkdown(content, state.viewerDocId);
        
        // 显示实际使用的配置
        if (extra.configUsed) {
            const cfg = extra.configUsed;
            html += `<div class="chat-config">调用: ${cfg.base_url} | 模型: ${cfg.model}</div>`;
        }
        
        if (extra.usage) {
            const u = extra.usage;
            html += `<div class="chat-usage">Token: ${u.total_tokens} (↑${u.prompt_tokens} / ↓${u.completion_tokens})</div>`;
        }
        msgDiv.innerHTML = html;
    } else {
        msgDiv.textContent = content;
    }
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

async function sendChat() {
    const query = dom.chatInput.value.trim();
    if (!query || !state.viewerDocId) return;
    
    if (state.indexStatus !== 'completed') {
        addChatMessage('error', '请先构建索引');
        return;
    }
    
    addChatMessage('user', query);
    dom.chatInput.value = '';
    dom.chatInput.disabled = true;
    dom.chatSendBtn.disabled = true;
    
    const loadingMsg = addChatMessage('loading', '正在思考中...');
    
    try {
        const startTime = Date.now();
        const resp = await fetch(`/api/documents/${state.viewerDocId}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });
        
        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(err);
        }
        
        const data = await resp.json();
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        
        let extraInfo = { usage: data.usage, elapsed: elapsed };
        
        // 显示实际使用的配置
        if (data.config_used) {
            const cfg = data.config_used;
            extraInfo.configUsed = cfg;
        }
        
        updateChatMessage(loadingMsg, 'assistant', data.answer, extraInfo);
    } catch (e) {
        updateChatMessage(loadingMsg, 'error', '问答失败: ' + e.message);
    } finally {
        dom.chatInput.disabled = false;
        dom.chatSendBtn.disabled = false;
        dom.chatInput.focus();
    }
}


// ============================================================
// Upload
// ============================================================

function showUploadModal() {
    dom.uploadModal.style.display = 'flex';
    dom.dropZone.style.display = 'block';
    dom.uploadProgress.style.display = 'none';
}

function hideUploadModal() {
    dom.uploadModal.style.display = 'none';
}

async function uploadFile(file) {
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
        alert('请选择 PDF 文件');
        return;
    }

    dom.dropZone.style.display = 'none';
    dom.uploadProgress.style.display = 'block';
    dom.uploadFileName.textContent = file.name;
    dom.uploadProgressBar.style.width = '0%';
    dom.uploadStatus.textContent = '上传中...';

    try {
        const formData = new FormData();
        formData.append('file', file);

        // 使用 XMLHttpRequest 以获取上传进度
        await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/upload');

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const pct = Math.round((e.loaded / e.total) * 100);
                    dom.uploadProgressBar.style.width = pct + '%';
                }
            };

            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    reject(new Error(`Upload failed: ${xhr.status}`));
                }
            };

            xhr.onerror = () => reject(new Error('Upload failed'));
            xhr.send(formData);
        });

        dom.uploadStatus.textContent = '上传成功！正在处理...';
        dom.uploadProgressBar.style.width = '100%';

        setTimeout(() => {
            hideUploadModal();
            loadDocuments();
        }, 800);

    } catch (e) {
        dom.uploadStatus.textContent = '上传失败: ' + e.message;
        dom.uploadProgressBar.style.width = '0%';
    }
}


// ============================================================
// Document Viewer
// ============================================================

async function openViewer(docId) {
    state.viewerDocId = docId;
    state.currentPage = 0;
    state.pageCache = {};
    state.chatHistory = [];
    state.chatVisible = false;

    try {
        state.viewerDoc = await api(`/api/documents/${docId}`);
    } catch (e) {
        alert('无法打开文档: ' + e.message);
        return;
    }

    dom.docTitle.textContent = state.viewerDoc.filename;
    dom.pageTotal.textContent = state.viewerDoc.page_count;
    dom.pageInput.max = state.viewerDoc.page_count;
    
    // 重置聊天
    dom.chatPane.style.display = 'none';
    dom.chatMessages.innerHTML = '';
    
    showView('viewer');
    loadPage(0);
    checkIndexStatus();

    // 如果文档还在处理中，轮询更新
    if (['queued', 'loading_model', 'processing'].includes(state.viewerDoc.status)) {
        startViewerPolling();
    }
}

let viewerPollInterval = null;

function startViewerPolling() {
    stopViewerPolling();
    viewerPollInterval = setInterval(async () => {
        if (!state.viewerDocId) return;
        try {
            state.viewerDoc = await api(`/api/documents/${state.viewerDocId}`);
            updateViewerStatus();
            if (state.viewerDoc.status === 'completed' || state.viewerDoc.status === 'error') {
                stopViewerPolling();
            }
        } catch {}
    }, 2000);
}

function stopViewerPolling() {
    if (viewerPollInterval) {
        clearInterval(viewerPollInterval);
        viewerPollInterval = null;
    }
}

function updateViewerStatus() {
    if (!state.viewerDoc) return;
    const doc = state.viewerDoc;
    const info = getStatusDisplay(doc);
    dom.viewerStatus.className = `badge ${info.badgeClass}`;
    if (doc.status === 'processing') {
        dom.viewerStatus.textContent = `${info.label} ${doc.processed_pages}/${doc.page_count}`;
    } else {
        dom.viewerStatus.textContent = info.label;
    }
}

async function loadPage(pageNum) {
    if (!state.viewerDoc) return;
    const doc = state.viewerDoc;

    // 边界检查
    pageNum = Math.max(0, Math.min(pageNum, doc.page_count - 1));
    state.currentPage = pageNum;

    // 更新页码 UI
    dom.pageInput.value = pageNum + 1;
    dom.prevPage.disabled = pageNum === 0;
    dom.nextPage.disabled = pageNum >= doc.page_count - 1;

    // 加载 PDF 页面图片
    dom.pdfPageImage.src = `/api/documents/${doc.id}/page/${pageNum}/pdf-image`;

    // 加载 Markdown 内容
    if (state.pageCache[pageNum] !== undefined) {
        dom.markdownBody.innerHTML = renderMathMarkdown(state.pageCache[pageNum], doc.id);
    } else {
        dom.markdownBody.innerHTML = '<div class="loading-placeholder"><div class="spinner"></div>加载中...</div>';
        try {
            const data = await api(`/api/documents/${doc.id}/page/${pageNum}/content`);
            state.pageCache[pageNum] = data.content;
            // 只有在当前页没变的情况下才更新
            if (state.currentPage === pageNum) {
                dom.markdownBody.innerHTML = renderMathMarkdown(data.content, doc.id);
            }
        } catch {
            if (state.currentPage === pageNum) {
                if (pageNum >= (doc.processed_pages || 0)) {
                    dom.markdownBody.innerHTML = '<div class="loading-placeholder"><div class="spinner"></div>该页正在处理中，请稍候...</div>';
                } else {
                    dom.markdownBody.innerHTML = '<div class="loading-placeholder">暂无内容</div>';
                }
            }
        }
    }

    // 预加载相邻页面
    prefetchPage(pageNum + 1);
    prefetchPage(pageNum - 1);

    updateViewerStatus();
}

async function prefetchPage(pageNum) {
    if (!state.viewerDoc) return;
    if (pageNum < 0 || pageNum >= state.viewerDoc.page_count) return;
    if (state.pageCache[pageNum] !== undefined) return;

    try {
        const data = await api(`/api/documents/${state.viewerDoc.id}/page/${pageNum}/content`);
        state.pageCache[pageNum] = data.content;
    } catch {
        // 忽略预取错误
    }
}


// ============================================================
// Resize Handle (拖拽分栏)
// ============================================================

function initResizeHandle() {
    const handle = dom.resizeHandle;
    const container = dom.viewerContent;
    const leftPane = dom.markdownPane;
    const rightPane = dom.pdfPane;
    let isResizing = false;

    handle.addEventListener('mousedown', (e) => {
        isResizing = true;
        handle.classList.add('active');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        const containerRect = container.getBoundingClientRect();
        const offset = e.clientX - containerRect.left;
        const totalWidth = containerRect.width;
        const pct = Math.max(20, Math.min(80, (offset / totalWidth) * 100));

        leftPane.style.flex = `0 0 ${pct}%`;
        rightPane.style.flex = `0 0 ${100 - pct}%`;
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            handle.classList.remove('active');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}


// ============================================================
// Event Binding
// ============================================================

function bindEvents() {
    // 上传
    dom.uploadBtn.addEventListener('click', showUploadModal);
    dom.modalClose.addEventListener('click', hideUploadModal);
    dom.uploadModal.addEventListener('click', (e) => {
        if (e.target === dom.uploadModal) hideUploadModal();
    });

    dom.dropZone.addEventListener('click', () => dom.fileInput.click());
    dom.fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) uploadFile(e.target.files[0]);
    });

    // 拖放
    dom.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dom.dropZone.classList.add('dragover');
    });
    dom.dropZone.addEventListener('dragleave', () => {
        dom.dropZone.classList.remove('dragover');
    });
    dom.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dom.dropZone.classList.remove('dragover');
        if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
    });

    // 查看器导航
    dom.backBtn.addEventListener('click', () => {
        stopViewerPolling();
        stopIndexPolling();
        state.viewerDocId = null;
        state.viewerDoc = null;
        state.pageCache = {};
        showView('library');
        loadDocuments();
    });

    dom.prevPage.addEventListener('click', () => {
        if (state.currentPage > 0) loadPage(state.currentPage - 1);
    });

    dom.nextPage.addEventListener('click', () => {
        if (state.viewerDoc && state.currentPage < state.viewerDoc.page_count - 1) {
            loadPage(state.currentPage + 1);
        }
    });

    dom.pageInput.addEventListener('change', () => {
        const val = parseInt(dom.pageInput.value, 10);
        if (!isNaN(val) && val >= 1 && state.viewerDoc && val <= state.viewerDoc.page_count) {
            loadPage(val - 1);
        } else {
            dom.pageInput.value = state.currentPage + 1;
        }
    });

    dom.pageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            dom.pageInput.blur();
            dom.pageInput.dispatchEvent(new Event('change'));
        }
    });

    // 键盘快捷键
    document.addEventListener('keydown', (e) => {
        if (state.currentView !== 'viewer') return;
        if (e.target.tagName === 'INPUT') return;

        if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
            e.preventDefault();
            if (state.currentPage > 0) loadPage(state.currentPage - 1);
        } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
            e.preventDefault();
            if (state.viewerDoc && state.currentPage < state.viewerDoc.page_count - 1) {
                loadPage(state.currentPage + 1);
            }
        } else if (e.key === 'Escape') {
            dom.backBtn.click();
        }
    });

    // 分栏拖拽
    initResizeHandle();
    
    // 设置模态框
    dom.settingsBtn.addEventListener('click', showSettingsModal);
    dom.settingsClose.addEventListener('click', hideSettingsModal);
    dom.settingsModal.addEventListener('click', (e) => {
        if (e.target === dom.settingsModal) hideSettingsModal();
    });
    dom.saveSettingsBtn.addEventListener('click', saveSettings);
    dom.providerSelect.addEventListener('change', (e) => onProviderChange(e.target.value));

    // 目录浏览器
    dom.browseModelPathBtn.addEventListener('click', browseDirectory);
    dom.browseClose.addEventListener('click', hideBrowseModal);
    dom.browseCancelBtn.addEventListener('click', hideBrowseModal);
    dom.browseModal.addEventListener('click', (e) => {
        if (e.target === dom.browseModal) hideBrowseModal();
    });
    dom.browseParentBtn.addEventListener('click', navigateToParent);
    dom.browseSelectBtn.addEventListener('click', selectCurrentDirectory);

    // RAG 功能
    dom.buildIndexBtn.addEventListener('click', buildIndex);
    dom.toggleChatBtn.addEventListener('click', toggleChat);
    dom.chatSendBtn.addEventListener('click', sendChat);
    dom.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChat();
        }
    });
}


// ============================================================
// 初始化
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    bindEvents();
    showView('library');
    loadDocuments();
    updateModelStatus();
    updateRagStatus();

    // 定期更新状态
    setInterval(updateModelStatus, 5000);
    setInterval(updateRagStatus, 10000);
});
