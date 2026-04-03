// Generate a unique session ID for the user
const sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
// currentSources is kept for backwards-compat but each message now tracks its own
let currentSources = [];
let isWaitingForResponse = false;

// DOM Elements
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const messagesContainer = document.getElementById('messages');
const roleSelect = document.getElementById('role-select');
const clearBtn = document.getElementById('clear-btn');
const modal = document.getElementById('source-modal');
const closeModalBtn = document.querySelector('.close-modal');
const modalBody = document.getElementById('modal-body');

// --- 1. Background Particles & Scroll Effects ---
const canvas = document.getElementById('particle-canvas');
const ctx = canvas.getContext('2d');
let particles = [];

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

class Particle {
    constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 2 + 0.5;
        this.baseSpeedY = Math.random() * -0.5 - 0.2; // Move slowly upwards
        this.speedY = this.baseSpeedY;
        this.speedX = (Math.random() - 0.5) * 0.5;
        this.opacity = Math.random() * 0.4 + 0.1;
    }
    update(scrollDelta) {
        // Adjust vertical position with momentum from scroll
        this.y += this.speedY - (scrollDelta * 0.15);
        this.x += this.speedX;

        // Wrap around
        if (this.y > canvas.height) this.y = 0;
        if (this.y < 0) this.y = canvas.height;
        if (this.x > canvas.width) this.x = 0;
        if (this.x < 0) this.x = canvas.width;
    }
    draw() {
        ctx.fillStyle = `rgba(0, 210, 255, ${this.opacity})`;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
}

// Init particles
for (let i = 0; i < 60; i++) {
    particles.push(new Particle());
}

let lastScrollY = window.scrollY;
let currentScrollDelta = 0;

window.addEventListener('scroll', () => {
    let delta = window.scrollY - lastScrollY;
    currentScrollDelta = delta;
    lastScrollY = window.scrollY;

    // Shift background hue subtly as user scrolls down
    const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
    const scrollPercent = Math.min(1, Math.max(0, window.scrollY / maxScroll));
    // Transitions from base dark navy (10, 14, 23) to slightly illuminated deep metallic blue (15, 25, 40)
    document.body.style.backgroundColor = `rgb(${10 + scrollPercent * 5}, ${14 + scrollPercent * 11}, ${23 + scrollPercent * 17})`;
});

function animateParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Smooth decay of scroll momentum
    currentScrollDelta *= 0.9;

    particles.forEach(p => {
        p.update(currentScrollDelta);
        p.draw();
    });
    requestAnimationFrame(animateParticles);
}
animateParticles();

// --- 2. Intersection Observer for Component Scroll Animations ---
document.addEventListener("DOMContentLoaded", () => {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
            }
        });
    }, { threshold: 0.1 });

    const fadeElement = document.querySelector('.fade-in-section');
    if (fadeElement) observer.observe(fadeElement);
});


// --- 2. Chat Functionality ---

// Handle standard form submission
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (isWaitingForResponse) return; // Prevent double submission

    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = '';
    await sendMessage(text);
});

// Helper for suggestion buttons — auto-submits just like Streamlit's st.button
window.sendSuggestion = function (text) {
    if (isWaitingForResponse) return; // Prevent clicking while generating
    chatInput.value = text;
    // Auto scroll down to chat if in mobile layout
    if (window.innerWidth <= 992) {
        document.querySelector('.chat-area').scrollIntoView({ behavior: 'smooth' });
    }
    // Trigger form submit so it sends immediately (matches old Streamlit behaviour)
    chatForm.requestSubmit();
}

// Clear Chat Function
clearBtn.addEventListener('click', async () => {
    try {
        await fetch('/api/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });

        // Reset UI
        messagesContainer.innerHTML = `
            <div class="message assistant-message intro-animation">
                <div class="avatar"><img src="/giphy.gif" alt="Bot" class="avatar-img"></div>
                <div class="content markdown-body">
                    <p>Hello! I am your <strong>Singapore Employment Law Assistant</strong>.</p>
                    <p>Chat cleared. How can I help you today?</p>
                </div>
            </div>
        `;

        // Hide context panel (facts are gone since memory was cleared)
        document.getElementById('context-panel').style.display = 'none';
        document.getElementById('context-text').textContent = '';
        currentSources = [];

    } catch (err) {
        console.error("Failed to clear chat", err);
    }
});

// Helper to toggle input state like Streamlit
function toggleInputs(disabled) {
    isWaitingForResponse = disabled;
    
    // Grey out the chatbar and change its placeholder
    chatInput.disabled = disabled;
    chatInput.style.opacity = disabled ? '0.5' : '1';
    chatInput.style.cursor = disabled ? 'not-allowed' : 'text';
    chatInput.placeholder = disabled ? 'Waiting for response...' : 'Ask about your employment rights...';
    
    const sendBtn = document.getElementById('send-btn');
    sendBtn.disabled = disabled;
    sendBtn.style.opacity = disabled ? '0.5' : '1';
    sendBtn.style.cursor = disabled ? 'not-allowed' : 'pointer';
    
    document.querySelectorAll('.suggestion-btn').forEach(btn => {
        btn.disabled = disabled;
        btn.style.opacity = disabled ? '0.5' : '1';
        btn.style.cursor = disabled ? 'not-allowed' : 'pointer';
    });
}

// Send message function integrating with FastAPI Backend
async function sendMessage(text) {
    // Disable inputs while generating
    toggleInputs(true);

    // 1. Add User Message to UI
    appendMessage('user', text);

    // 2. Add Typing Indicator to UI
    const typingId = appendTypingIndicator();

    // Scroll to bottom
    scrollToBottom();

    try {
        // 3. Make API Call
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                session_id: sessionId,
                role: roleSelect.value
            })
        });

        const data = await response.json();

        // 4. Remove Typing Indicator
        document.getElementById(typingId)?.remove();

        if (!response.ok) {
            throw new Error(data.detail || "Server error");
        }

        // Store sources for modal lookup (keep global updated too for compat)
        if (data.sources && data.sources.length > 0) {
            currentSources = data.sources;
        }

        // 5. Append Assistant Response to UI (pass sources directly so each message owns its data)
        appendMessage('assistant', data.reply, data.warnings, data.confidence, data.sources || []);

        // 6. Update sidebar context panel (mirrors Streamlit's "Detected context" section)
        updateContextPanel();

    } catch (err) {
        // Use optional chaining — if the typing indicator was already removed inside the
        // try block (e.g. on a !response.ok error), getElementById returns null and
        // calling .remove() on null would throw, swallowing the error message entirely.
        document.getElementById(typingId)?.remove();
        appendMessage('assistant', `**Error:** Could not connect to the server. ${err.message}`);
    } finally {
        // Re-enable inputs once done
        toggleInputs(false);
        chatInput.focus();
    }

    scrollToBottom();
}

// Fetch and display extracted user context facts in the sidebar
async function updateContextPanel() {
    try {
        const res = await fetch(`/api/context?session_id=${sessionId}`);
        const data = await res.json();
        const panel = document.getElementById('context-panel');
        const contextText = document.getElementById('context-text');

        if (data.context && data.context.trim()) {
            // Format "salary: $3,200; job type: clerk" → bullet list
            const lines = data.context.split(';').map(f => '• ' + f.trim()).join('\n');
            contextText.textContent = lines;
            panel.style.display = 'block';
        } else {
            panel.style.display = 'none';
        }
    } catch (e) {
        // Non-critical — silently ignore if context fetch fails
    }
}

function appendMessage(role, text, warnings = [], confidence = null, sources = []) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;

    let avatarContent = role === 'user'
        ? '<img src="/assets/81fb5188c7eba22df8715916a17f4c54-ezgif.com-rotate.gif" alt="User" class="avatar-img">'
        : '<img src="/giphy.gif" alt="Bot" class="avatar-img">';

    // Parse Markdown if assistant
    let parsedContent = role === 'assistant' ? marked.parse(text) : text;

    let innerHTML = `
        <div class="avatar">${avatarContent}</div>
        <div class="content ${role === 'assistant' ? 'markdown-body' : ''}">
            ${parsedContent}
    `;

    // Append warnings if any
    if (warnings && warnings.length > 0) {
        warnings.forEach(w => {
            innerHTML += `<div class="warning-alert"><i class="fa-solid fa-triangle-exclamation"></i> ${w}</div>`;
        });
    }

    // Append Confidence & Source Button if assistant
    if (role === 'assistant' && confidence) {
        let colorClass = '#10b981'; // green
        if (confidence.color === 'orange') colorClass = '#f59e0b';
        if (confidence.color === 'red') colorClass = '#ef4444';

        innerHTML += `
            <div class="confidence-badge" style="color: ${colorClass}; border: 1px solid ${colorClass};">
                <i class="fa-solid fa-bullseye"></i> Confidence: ${confidence.label} (${confidence.score}/100)
            </div>
        `;
    }

    // Each message captures its own sources — clicking the button shows that message's sources
    const hasSources = Array.isArray(sources) && sources.length > 0;
    if (role === 'assistant' && hasSources) {
        // Store sources as a data attribute (JSON) so older messages still work correctly
        const encodedSources = encodeURIComponent(JSON.stringify(sources));
        innerHTML += `
            <div style="margin-top: 10px;">
                <button class="glass-btn source-button" onclick="openModalWithSources(decodeURIComponent('${encodedSources}'))">
                    <i class="fa-solid fa-book-open"></i> View Retrieved Sources
                </button>
            </div>
        `;
    }

    innerHTML += `</div>`;
    msgDiv.innerHTML = innerHTML;

    messagesContainer.appendChild(msgDiv);
}

function appendTypingIndicator() {
    const id = 'typing-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = `message assistant-message`;
    msgDiv.id = id;

    msgDiv.innerHTML = `
        <div class="avatar"><img src="/giphy.gif" alt="Bot" class="avatar-img"></div>
        <div class="content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    messagesContainer.appendChild(msgDiv);
    return id;
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// --- 3. Modal Functionality ---

// Opens the modal with sources passed directly from a specific message
window.openModalWithSources = function (sourcesJson) {
    const sources = JSON.parse(sourcesJson);
    currentSources = sources; // keep global in sync
    _renderModal(sources);
}

// Fallback: opens modal with the global (latest) sources
window.openModal = function () {
    _renderModal(currentSources);
}

function _renderModal(sources) {
    modalBody.innerHTML = '';

    if (!sources || sources.length === 0) {
        modalBody.innerHTML = '<p>No sources were retrieved for this query.</p>';
    } else {
        sources.forEach(src => {
            const div = document.createElement('div');
            div.className = 'source-item';
            div.innerHTML = `
                <div class="source-meta">[${src.index}] ${src.type}: ${src.label} (Score: ${parseFloat(src.score).toFixed(4)})</div>
                ${src.url ? `<p><a href="${src.url}" target="_blank" style="color:var(--accent-primary)">View Online Document</a></p><br>` : ''}
                <div class="source-text">${src.text}</div>
            `;
            modalBody.appendChild(div);
        });
    }

    modal.style.display = 'flex';
}

closeModalBtn.onclick = function () {
    modal.style.display = 'none';
}

window.onclick = function (event) {
    if (event.target == modal) {
        modal.style.display = 'none';
    }
}
