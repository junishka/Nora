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
  // Render markdown — Claude emits **bold**, `code`, lists, fenced
  // code blocks, etc. BuilderMarkdown escapes HTML before injecting
  // specific tags, so assistant-generated HTML / scripts never
  // execute. See src/builder/web/markdown.js.
  append('assistant', text, /*markdown=*/ true);
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

function append(kind, text, markdown) {
  const wrapper = document.createElement('div');
  wrapper.className = 'message ' + kind;
  const body = document.createElement('div');
  body.className = 'message-body';
  if (markdown && window.BuilderMarkdown) {
    body.innerHTML = window.BuilderMarkdown.render(text);
  } else {
    body.textContent = text;
  }
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
  const existingCard = [...messagesEl.querySelectorAll('.tool-card')]
    .find((c) => c.dataset.callId === evt.call_id);

  // For submit_script / expand_result (anything with raw R/Stata
  // output), render the native R / Stata output prominently BEFORE
  // the sanitized JSON. The researcher recognizes the regression
  // table; the JSON is secondary. Matches the terminal TUI's split.
  const hasRawOutput = !!(evt.raw_stdout || evt.raw_stderr);

  if (existingCard) {
    if (evt.is_error) {
      existingCard.classList.add('error');
    }
    const statusEl = existingCard.querySelector('.tool-status');
    if (statusEl) statusEl.textContent = evt.is_error ? 'error' : 'done';
    const body = existingCard.querySelector('.tool-body');

    if (hasRawOutput) {
      // Keep the tool-call card compact and put the output as its
      // own panel right after. The call card expands on click for
      // audit purposes; the primary visual is the output panel.
      existingCard.classList.add('collapsed');
      messagesEl.appendChild(buildResultPanel(evt));
    } else {
      // Non-script tool (get_schema, request_data, list_results) —
      // the JSON IS the useful output. Keep it inline in the card.
      const resultPre = document.createElement('pre');
      resultPre.textContent = prettyJson(evt.text);
      body.appendChild(resultPre);
    }
    scrollToBottom();
  } else {
    // No matching card — render a standalone panel.
    if (hasRawOutput) {
      messagesEl.appendChild(buildResultPanel(evt));
    } else {
      const card = document.createElement('div');
      card.className = 'tool-card' + (evt.is_error ? ' error' : '');
      const pre = document.createElement('pre');
      pre.textContent = prettyJson(evt.text);
      card.appendChild(pre);
      messagesEl.appendChild(card);
    }
    scrollToBottom();
  }
}

function buildResultPanel(evt) {
  /* Result panel for submit_script / expand_result events — the ones
   * that have raw R/Stata output. Layout:
   *   ┌─ R / Stata output (always visible) ─────────┐
   *   │  <pre>...native regression table...</pre>    │
   *   ├─ stderr (if present, yellow-tinted) ────────┤
   *   │  <pre>warnings...</pre>                      │
   *   ├─ Sanitized output (collapsed, toggle) ──────┤
   *   │  <pre>{ ... clamped JSON ... }</pre>         │
   *   └──────────────────────────────────────────────┘ */
  const panel = document.createElement('div');
  panel.className = 'result-panel' + (evt.is_error ? ' error' : '');

  const stdoutSection = document.createElement('section');
  stdoutSection.className = 'result-stdout';
  const stdoutHeader = document.createElement('div');
  stdoutHeader.className = 'result-header';
  stdoutHeader.textContent = 'R / Stata output';
  stdoutSection.appendChild(stdoutHeader);
  const stdoutPre = document.createElement('pre');
  stdoutPre.textContent = (evt.raw_stdout || '').trimEnd() || '(no output)';
  stdoutSection.appendChild(stdoutPre);
  panel.appendChild(stdoutSection);

  if (evt.raw_stderr && evt.raw_stderr.trim()) {
    const stderrSection = document.createElement('section');
    stderrSection.className = 'result-stderr';
    const stderrHeader = document.createElement('div');
    stderrHeader.className = 'result-header';
    stderrHeader.textContent = 'stderr';
    stderrSection.appendChild(stderrHeader);
    const stderrPre = document.createElement('pre');
    stderrPre.textContent = evt.raw_stderr.trimEnd();
    stderrSection.appendChild(stderrPre);
    panel.appendChild(stderrSection);
  }

  const sanitizedSection = document.createElement('section');
  sanitizedSection.className = 'result-sanitized collapsed';
  const sanitizedHeader = document.createElement('div');
  sanitizedHeader.className = 'result-header result-toggle';
  const arrow = document.createElement('span');
  arrow.className = 'toggle-arrow';
  arrow.textContent = '▼';
  const label = document.createElement('span');
  label.textContent = 'Sanitized output (what Claude saw)';
  sanitizedHeader.appendChild(arrow);
  sanitizedHeader.appendChild(label);
  sanitizedSection.appendChild(sanitizedHeader);
  const sanitizedPre = document.createElement('pre');
  sanitizedPre.textContent = prettyJson(evt.text);
  sanitizedSection.appendChild(sanitizedPre);
  sanitizedHeader.addEventListener('click', () => {
    sanitizedSection.classList.toggle('collapsed');
  });
  panel.appendChild(sanitizedSection);

  if (evt.run_dir) {
    const note = document.createElement('div');
    note.className = 'result-note';
    note.textContent = 'Full log: ' + evt.run_dir;
    panel.appendChild(note);
  }

  return panel;
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
