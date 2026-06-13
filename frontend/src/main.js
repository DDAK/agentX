/**
 * AgentX frontend — WebSocket chat client.
 *
 * Architecture
 * ────────────
 *  SessionStore   — fetches / caches sessions from the REST API
 *  AgentSocket    — manages the WebSocket connection to /api/sessions/:id/ws
 *  UI             — renders messages, handles input, connects the two above
 */
import { marked } from 'marked'

// ── marked config ─────────────────────────────────────────────────────────────
marked.setOptions({ breaks: true, gfm: true })

// ── constants ─────────────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_BASE ?? ''

// ── SessionStore ──────────────────────────────────────────────────────────────

const SessionStore = {
  /** @type {Session[]} */
  sessions: [],

  async fetchAll() {
    const r = await fetch(`${API_BASE}/api/sessions`)
    if (!r.ok) throw new Error(`Failed to list sessions: ${r.status}`)
    this.sessions = await r.json()
    return this.sessions
  },

  async create(label = null) {
    const r = await fetch(`${API_BASE}/api/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    })
    if (!r.ok) throw new Error(`Failed to create session: ${r.status}`)
    const s = await r.json()
    this.sessions.unshift(s)
    return s
  },

  async get(id) {
    const r = await fetch(`${API_BASE}/api/sessions/${id}`)
    if (!r.ok) return null
    return r.json()
  },
}

// ── AgentSocket ───────────────────────────────────────────────────────────────

/**
 * Manages a single WebSocket connection to /api/sessions/:id/ws.
 * Emits events:
 *   onEvent(event: AgentEvent)  — every server event
 *   onOpen()
 *   onClose()
 *   onError(e)
 */
class AgentSocket {
  constructor(sessionId) {
    this.sessionId = sessionId
    this.ws = null
    this.onEvent = () => {}
    this.onOpen  = () => {}
    this.onClose = () => {}
    this.onError = () => {}
  }

  get wsBase() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host  = API_BASE ? new URL(API_BASE).host : location.host
    return `${proto}//${host}`
  }

  connect() {
    if (this.ws) this.disconnect()

    const url = `${this.wsBase}/api/sessions/${this.sessionId}/ws`
    this.ws = new WebSocket(url)

    this.ws.onopen    = () => this.onOpen()
    this.ws.onclose   = () => this.onClose()
    this.ws.onerror   = (e) => this.onError(e)
    this.ws.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data)
        this.onEvent(event)
      } catch {
        console.warn('unparseable WS frame', e.data)
      }
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.onclose = null // suppress onClose during intentional disconnect
      this.ws.close()
      this.ws = null
    }
  }

  send(text) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ text }))
    }
  }
}

// ── UI state ──────────────────────────────────────────────────────────────────

const state = {
  activeSessionId: null,
  socket: null,
  /** @type {HTMLElement|null} — current streaming agent message bubble */
  streamingBubble: null,
}

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dom = {
  sessionList:   document.getElementById('session-list'),
  messages:      document.getElementById('messages'),
  inputForm:     document.getElementById('input-form'),
  messageInput:  document.getElementById('message-input'),
  btnSend:       document.getElementById('btn-send'),
  btnNewSession: document.getElementById('btn-new-session'),
  connStatus:    document.getElementById('conn-status'),
  connLabel:     document.getElementById('conn-label'),
  sessionLabel:  document.getElementById('session-label-display'),
  typingIndicator: document.getElementById('typing-indicator'),
}

// ── connection status ─────────────────────────────────────────────────────────

function setStatus(status) {
  dom.connStatus.className = `status-dot ${status}`
  dom.connLabel.textContent = {
    connected:    'Connected',
    connecting:   'Connecting…',
    disconnected: 'Disconnected',
  }[status] ?? status
  dom.btnSend.disabled = status !== 'connected'
}

// ── render sessions sidebar ───────────────────────────────────────────────────

function renderSessions(sessions) {
  dom.sessionList.innerHTML = ''
  for (const s of sessions) {
    const li = document.createElement('li')
    if (s.id === state.activeSessionId) li.classList.add('active')

    const label = document.createElement('span')
    label.className = 'session-item-label'
    label.textContent = s.label || `Session ${s.id.slice(0, 8)}`

    const meta = document.createElement('span')
    meta.className = 'session-item-meta'
    meta.textContent = `${s.message_count} messages · ${timeAgo(s.updated_at)}`

    li.append(label, meta)
    li.addEventListener('click', () => openSession(s.id))
    dom.sessionList.appendChild(li)
  }
}

// ── open / switch session ─────────────────────────────────────────────────────

async function openSession(id) {
  if (state.activeSessionId === id) return

  // Disconnect previous socket.
  if (state.socket) {
    state.socket.disconnect()
    state.socket = null
  }

  state.activeSessionId = id
  state.streamingBubble = null
  dom.messages.innerHTML = ''
  setStatus('connecting')

  // Load existing history.
  const session = await SessionStore.get(id)
  if (session) {
    dom.sessionLabel.textContent = session.label || `Session ${id.slice(0, 8)}`
    // Replay non-system messages.
    for (const msg of session.messages) {
      if (msg.role === 'system') continue
      if (msg.role === 'user') {
        appendMessage('user', extractText(msg.content))
      } else if (msg.role === 'assistant' && extractText(msg.content)) {
        appendMessage('agent', extractText(msg.content))
      }
    }
  }

  // Refresh sidebar active state.
  renderSessions(SessionStore.sessions)

  // Open WebSocket.
  const sock = new AgentSocket(id)
  state.socket = sock

  sock.onOpen  = () => {
    setStatus('connected')
    scrollToBottom()
  }
  sock.onClose = () => {
    setStatus('disconnected')
    state.streamingBubble = null
  }
  sock.onError = () => setStatus('disconnected')

  sock.onEvent = (event) => handleAgentEvent(event)

  sock.connect()
}

// ── handle incoming agent events ──────────────────────────────────────────────

function handleAgentEvent(event) {
  switch (event.type) {
    case 'thinking': {
      dom.typingIndicator.hidden = false
      scrollToBottom()
      break
    }

    case 'text': {
      dom.typingIndicator.hidden = true
      // Append to (or start) a streaming bubble.
      if (!state.streamingBubble) {
        state.streamingBubble = appendMessage('agent', '')
      }
      // Accumulate raw text, then re-render markdown.
      state.streamingBubble.dataset.raw = (state.streamingBubble.dataset.raw || '') + event.text
      state.streamingBubble.innerHTML = marked.parse(state.streamingBubble.dataset.raw)
      scrollToBottom()
      break
    }

    case 'tool_call': {
      appendToolEvent('call', event.name, JSON.stringify(event.input))
      break
    }

    case 'tool_result': {
      const preview = event.result.length > 120
        ? event.result.slice(0, 120) + '…'
        : event.result
      appendToolEvent('result', event.name, preview)
      break
    }

    case 'iteration_limit_reached': {
      appendSystemMsg('⚠ Iteration limit reached.')
      break
    }

    case 'error': {
      appendSystemMsg(`⚠ Error: ${event.message}`)
      break
    }

    case 'turn_done': {
      dom.typingIndicator.hidden = true
      state.streamingBubble = null
      // Refresh session list (message count updated).
      SessionStore.fetchAll().then(renderSessions).catch(() => {})
      scrollToBottom()
      break
    }
  }
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

/** Append a user/agent message bubble. Returns the .msg-body element. */
function appendMessage(role, text) {
  const wrap = document.createElement('div')
  wrap.className = `msg ${role}`

  const roleEl = document.createElement('div')
  roleEl.className = `msg-role ${role}`
  roleEl.textContent = role === 'user' ? 'You' : 'AgentX'

  const body = document.createElement('div')
  body.className = 'msg-body'

  if (role === 'user') {
    body.textContent = text
  } else {
    body.dataset.raw = text
    body.innerHTML   = text ? marked.parse(text) : ''
  }

  wrap.append(roleEl, body)
  dom.messages.appendChild(wrap)
  scrollToBottom()
  return body
}

/** Append a tool call or result row. */
function appendToolEvent(kind, name, detail) {
  const row = document.createElement('div')
  row.className = `tool-event ${kind}`

  const icon = document.createElement('span')
  icon.className = 'tool-icon'
  icon.textContent = kind === 'call' ? '⚙' : '✓'

  const nameEl = document.createElement('span')
  nameEl.className = 'tool-name'
  nameEl.textContent = name

  const detailEl = document.createElement('span')
  detailEl.className = 'tool-detail'
  detailEl.textContent = detail

  row.append(icon, nameEl, detailEl)
  dom.messages.appendChild(row)
  scrollToBottom()
}

function appendSystemMsg(text) {
  const el = document.createElement('div')
  el.className = 'msg system'
  el.textContent = text
  dom.messages.appendChild(el)
  scrollToBottom()
}

function scrollToBottom() {
  dom.messages.scrollTop = dom.messages.scrollHeight
}

// ── send message ──────────────────────────────────────────────────────────────

function sendMessage(text) {
  if (!text.trim() || !state.socket) return
  appendMessage('user', text)
  state.socket.send(text)
  dom.messageInput.value = ''
  dom.messageInput.style.height = 'auto'
}

// ── input form events ─────────────────────────────────────────────────────────

dom.inputForm.addEventListener('submit', (e) => {
  e.preventDefault()
  sendMessage(dom.messageInput.value)
})

dom.messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage(dom.messageInput.value)
  }
})

// Auto-resize textarea.
dom.messageInput.addEventListener('input', () => {
  dom.messageInput.style.height = 'auto'
  dom.messageInput.style.height = `${dom.messageInput.scrollHeight}px`
})

// ── new session ───────────────────────────────────────────────────────────────

dom.btnNewSession.addEventListener('click', async () => {
  const label = prompt('Session label (optional):')
  try {
    const session = await SessionStore.create(label || null)
    renderSessions(SessionStore.sessions)
    await openSession(session.id)
  } catch (e) {
    alert(`Failed to create session: ${e.message}`)
  }
})

// ── utilities ─────────────────────────────────────────────────────────────────

function extractText(content) {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    const part = content.find(p => p.type === 'text')
    return part?.text ?? ''
  }
  return ''
}

function timeAgo(isoString) {
  const diff = Date.now() - new Date(isoString).getTime()
  const m = Math.floor(diff / 60_000)
  if (m < 1)   return 'just now'
  if (m < 60)  return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24)  return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

// ── initialise ────────────────────────────────────────────────────────────────

async function init() {
  setStatus('disconnected')

  try {
    const sessions = await SessionStore.fetchAll()
    renderSessions(sessions)

    if (sessions.length > 0) {
      await openSession(sessions[0].id)
    } else {
      // Auto-create a first session.
      const s = await SessionStore.create('First session')
      renderSessions(SessionStore.sessions)
      await openSession(s.id)
    }
  } catch (e) {
    console.error('Init failed:', e)
    appendSystemMsg(`⚠ Could not reach AgentX backend: ${e.message}`)
  }
}

init()
