/* Builder web UI — frontend JS.
 *
 * Communicates with the Python backend via pywebview's bridge:
 *   window.pywebview.api.<method>(...)    — JS → Python
 *   Python calls `window.builder_event({type, ...payload})` via
 *     webview.evaluate_js to push events back to us.
 *
 * This file deliberately avoids a framework. The chat state is the
 * DOM; each event appends a message node to #messages. Swap for a
 * real framework later if we want components and state management.
 */

const messagesEl = document.getElementById('messages');
const form = document.getElementById('compose-form');
const input = document.getElementById('compose-input');
const sendBtn = document.getElementById('send-btn');
const welcomeEl = document.getElementById('welcome');
const cwdEl = document.getElementById('cwd-display');
const policyEl = document.getElementById('policy-summary');

// ----- send / receive -----------------------------------------------------

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  if (!window.pywebview || !window.pywebview.api) {
    appendSystem('backend not ready yet; try again');
    return;
  }
  appendUser(text);
  input.value = '';
  autosize();
  setSending(true);
  try {
    await window.pywebview.api.send_message(text);
  } catch (err) {
    appendError('send failed: ' + err);
  } finally {
    setSending(false);
  }
});

// Shift-Enter inserts a newline; plain Enter sends.
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    form.dispatchEvent(new Event('submit'));
  }
});

// Auto-grow textarea.
input.addEventListener('input', autosize);
function autosize() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 220) + 'px';
}

function setSending(sending) {
  sendBtn.disabled = sending;
  sendBtn.textContent = sending ? '…' : 'Send';
}

// ----- Python → JS event handler -----------------------------------------

window.builder_event = function (evt) {
  // evt is a plain object; {type} + type-specific fields.
  switch (evt.type) {
    case 'ready':
      welcomeEl.textContent = evt.greeting || 'Ready. Ask a question about your data.';
      cwdEl.textContent = evt.cwd || '';
      updatePolicySummary(evt.policy);
      break;
    case 'assistant_text':
      appendAssistant(evt.text);
      break;
    case 'assistant_thinking':
      appendThinking(evt.text);
      break;
    case 'tool_call':
      appendToolCall(evt);
      break;
    case 'tool_result':
      appendToolResult(evt);
      break;
    case 'turn_done':
      // Could show token count later; for now silent.
      break;
    case 'auth_failure':
      appendError('Auth failure: ' + (evt.reason || 'unknown'));
      break;
    case 'turn_error':
      appendError(evt.message || 'unknown error');
      break;
    case 'policy_updated':
      updatePolicySummary(evt.policy);
      break;
    default:
      console.warn('unknown event type', evt);
  }
};

// ----- message rendering --------------------------------------------------

function appendUser(text) {
  append('user', text);
}

function appendAssistant(text) {
  append('assistant', text);
}

function appendThinking(text) {
  append('thinking', text);
}

function appendSystem(text) {
  append('system', text);
}

function appendError(text) {
  append('error', text);
}

function append(kind, text) {
  const wrapper = document.createElement('div');
  wrapper.className = 'message ' + kind;
  const body = document.createElement('div');
  body.className = 'message-body';
  body.textContent = text;
  wrapper.appendChild(body);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
}

function appendToolCall(evt) {
  const card = document.createElement('div');
  card.className = 'tool-card collapsed';
  card.dataset.callId = evt.call_id;

  const header = document.createElement('div');
  header.className = 'tool-header';
  const arrow = document.createElement('span');
  arrow.className = 'tool-arrow';
  arrow.textContent = '▼';
  const title = document.createElement('span');
  title.innerHTML =
    '<span class="tool-name">' + shortenToolName(evt.name) + '</span>' +
    ' <span class="tool-status">running…</span>';
  header.appendChild(arrow);
  header.appendChild(title);

  const body = document.createElement('div');
  body.className = 'tool-body';
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(evt.input, null, 2);
  body.appendChild(pre);

  header.addEventListener('click', () => card.classList.toggle('collapsed'));

  card.appendChild(header);
  card.appendChild(body);
  messagesEl.appendChild(card);
  scrollToBottom();
}

function appendToolResult(evt) {
  // Find the matching tool-call card and append the result to it.
  const existingCard = [...messagesEl.querySelectorAll('.tool-card')]
    .find((c) => c.dataset.callId === evt.call_id);
  if (existingCard) {
    if (evt.is_error) {
      existingCard.classList.add('error');
    }
    const statusEl = existingCard.querySelector('.tool-status');
    if (statusEl) statusEl.textContent = evt.is_error ? 'error' : 'done';

    const body = existingCard.querySelector('.tool-body');
    const resultPre = document.createElement('pre');
    resultPre.textContent = prettyJson(evt.text);
    body.appendChild(resultPre);

    if (evt.run_dir) {
      const rawNote = document.createElement('div');
      rawNote.style.fontSize = '11px';
      rawNote.style.color = 'var(--text-dim)';
      rawNote.textContent = 'Raw R/Stata output: ' + evt.run_dir + '/stdout.log';
      body.appendChild(rawNote);
    }
    scrollToBottom();
  } else {
    // No matching card (shouldn't happen often) — render standalone.
    const card = document.createElement('div');
    card.className = 'tool-card' + (evt.is_error ? ' error' : '');
    const pre = document.createElement('pre');
    pre.textContent = prettyJson(evt.text);
    card.appendChild(pre);
    messagesEl.appendChild(card);
    scrollToBottom();
  }
}

function shortenToolName(name) {
  // mcp__builder__submit_script → submit_script
  const parts = name.split('__');
  return parts[parts.length - 1] || name;
}

function prettyJson(text) {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch (_) {
    return text;
  }
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function updatePolicySummary(policy) {
  // Reset whatever was here — this function is called on initial load
  // and whenever the policy changes.
  policyEl.innerHTML = '';
  if (!policy) {
    policyEl.textContent = '(no datasets)';
    return;
  }
  const datasets = policy.datasets || [];
  if (datasets.length === 0) {
    policyEl.textContent = 'no datasets in cwd';
    return;
  }

  const n = datasets.length;
  const customized = datasets.filter((d) => d.explicit).length;
  const defaultDepth = policy.default_max_depth;
  const plural = n === 1 ? '' : 's';

  let summary;
  if (customized === 0) {
    summary = `${n} dataset${plural} · all at ${defaultDepth} (default)`;
  } else if (customized === n) {
    summary = `${n} datasets · all have custom ceilings`;
  } else {
    summary = `${n} datasets · ${customized} customized, rest at ${defaultDepth}`;
  }

  const summarySpan = document.createElement('span');
  summarySpan.textContent = `Schema policy: ${summary}`;
  policyEl.appendChild(summarySpan);

  const toggle = document.createElement('a');
  toggle.href = '#';
  toggle.className = 'policy-toggle';
  toggle.textContent = 'show all';
  policyEl.appendChild(toggle);

  const details = document.createElement('div');
  details.className = 'policy-details hidden';
  datasets.forEach((d) => {
    const row = document.createElement('div');
    row.textContent =
      `${d.name} → ${d.ceiling}${d.explicit ? '' : ' (default)'}`;
    details.appendChild(row);
  });
  policyEl.appendChild(details);

  toggle.addEventListener('click', (e) => {
    e.preventDefault();
    details.classList.toggle('hidden');
    toggle.textContent = details.classList.contains('hidden') ? 'show all' : 'hide';
  });
}

// Signal to the Python side that we're ready to receive events. pywebview
// sets window.pywebview once its bridge is ready; until then we wait.
function whenReady(fn) {
  if (window.pywebview && window.pywebview.api) return fn();
  window.addEventListener('pywebviewready', fn, { once: true });
}

whenReady(() => {
  // Ask the backend for initial state (cwd, policy, etc.).
  window.pywebview.api.ui_ready();
});
