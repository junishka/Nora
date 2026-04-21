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

const landingEl = document.getElementById('landing');
const chatEl = document.getElementById('chat');
const dropZone = document.getElementById('drop-zone');
const chooseFilesBtn = document.getElementById('choose-files-btn');
const chooseFolderBtn = document.getElementById('choose-folder-btn');
const landingStatus = document.getElementById('landing-status');
const messagesEl = document.getElementById('messages');
const form = document.getElementById('compose-form');
const input = document.getElementById('compose-input');
const sendBtn = document.getElementById('send-btn');
const welcomeEl = document.getElementById('welcome');
const cwdEl = document.getElementById('cwd-display');

// Depth tiers — kept in sync with builder/policy.py::VALID_DEPTHS and
// with the get_schema tool help. Displayed labels are researcher-
// facing — plain English, not internal identifiers.
const DEPTH_TIERS = [
  { value: 'names_only',                 label: 'Names only' },
  { value: 'names_types',                label: 'Names + types (default)' },
  { value: 'names_types_labels',         label: '+ labels / value labels' },
  { value: 'names_types_labels_summary', label: '+ NA count / distinct count' },
];

// ----- view routing ------------------------------------------------------

function showLanding() {
  landingEl.classList.remove('hidden');
  chatEl.classList.add('hidden');
}

function showChat(payload) {
  landingEl.classList.add('hidden');
  chatEl.classList.remove('hidden');
  welcomeEl.textContent = payload.greeting || 'Ready.';
  cwdEl.textContent = payload.cwd || '';
  // Remove any prior policy card so a re-show doesn't duplicate it
  // (e.g., if a future feature re-renders the chat view).
  const prior = document.getElementById('policy-card');
  if (prior) prior.remove();
  const card = buildPolicyCard(payload.policy);
  if (card) {
    // Insert above the welcome so it's the first thing the
    // researcher sees.
    messagesEl.insertBefore(card, messagesEl.firstChild);
  }
  input.focus();
}

// ----- landing: file picker / folder picker / drag-drop -----------------

function setLandingBusy(busy, msg) {
  chooseFilesBtn.disabled = busy;
  chooseFolderBtn.disabled = busy;
  landingStatus.classList.remove('error');
  landingStatus.textContent = msg || '';
}

function setLandingError(msg) {
  chooseFilesBtn.disabled = false;
  chooseFolderBtn.disabled = false;
  landingStatus.classList.add('error');
  landingStatus.textContent = msg;
}

async function handleSessionResult(result) {
  if (!result) {
    setLandingError('no response from the backend');
    return;
  }
  if (!result.ok) {
    const reason = result.reason || 'unknown';
    if (reason === 'cancelled') {
      // Researcher cancelled the dialog — no noise, just clear status.
      setLandingBusy(false, '');
    } else {
      setLandingError(reason);
    }
    return;
  }
  showChat(result);
}

chooseFilesBtn.addEventListener('click', async () => {
  if (!window.pywebview || !window.pywebview.api) return;
  setLandingBusy(true, 'Opening file picker…');
  try {
    const result = await window.pywebview.api.choose_files();
    await handleSessionResult(result);
  } catch (err) {
    setLandingError('failed: ' + err);
  }
});

chooseFolderBtn.addEventListener('click', async () => {
  if (!window.pywebview || !window.pywebview.api) return;
  setLandingBusy(true, 'Opening folder picker…');
  try {
    const result = await window.pywebview.api.choose_folder();
    await handleSessionResult(result);
  } catch (err) {
    setLandingError('failed: ' + err);
  }
});

// Drag-drop. Visual state on the drop zone; actual handling on the whole
// landing area so a near-miss still works.
['dragenter', 'dragover'].forEach((name) => {
  landingEl.addEventListener(name, (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
});
['dragleave', 'drop'].forEach((name) => {
  landingEl.addEventListener(name, (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
  });
});

landingEl.addEventListener('drop', async (e) => {
  e.preventDefault();
  const dt = e.dataTransfer;
  if (!dt || !dt.files || dt.files.length === 0) return;
  const files = Array.from(dt.files);
  // Only data files we recognize. Anything else gets rejected up-front
  // so researchers see the reason in the UI rather than a confused
  // session with non-data files in it.
  const accepted = files.filter((f) => /\.(csv|dta|rds)$/i.test(f.name));
  if (accepted.length === 0) {
    setLandingError(
      'Drop .csv, .dta, or .rds files. Other types are ignored.'
    );
    return;
  }
  setLandingBusy(true, `Uploading ${accepted.length} file${accepted.length === 1 ? '' : 's'}…`);
  try {
    const payload = await Promise.all(accepted.map(readFileAsBase64));
    const result = await window.pywebview.api.upload_files(payload);
    await handleSessionResult(result);
  } catch (err) {
    setLandingError('upload failed: ' + err);
  }
});

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // result is a data URL; base64 content is after the first comma.
      const dataUrl = reader.result;
      const comma = dataUrl.indexOf(',');
      resolve({
        name: file.name,
        content: comma >= 0 ? dataUrl.substring(comma + 1) : dataUrl,
      });
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

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
      showChat(evt);
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
    // Action buttons — opt-in "show me the raw result in my native
    // app" affordances. None of these fire automatically; researcher
    // clicks what they want.
    const actions = document.createElement('div');
    actions.className = 'result-actions';

    const openOutputBtn = document.createElement('button');
    openOutputBtn.type = 'button';
    openOutputBtn.className = 'result-action';
    openOutputBtn.textContent = 'Open output';
    openOutputBtn.title =
      'Open the full R/Stata stdout log in your default text editor.';
    openOutputBtn.addEventListener('click', () =>
      openInNativeApp(evt.run_dir + '/stdout.log', openOutputBtn)
    );
    actions.appendChild(openOutputBtn);

    const openScriptBtn = document.createElement('button');
    openScriptBtn.type = 'button';
    openScriptBtn.className = 'result-action';
    openScriptBtn.textContent = 'Open script';
    openScriptBtn.title =
      'Open the R or Stata script in its default app (RStudio / Stata). '
      + 'You can re-run it there to see the native output yourself.';
    openScriptBtn.addEventListener('click', () => {
      // Try script.R first; if not present try script.do. The bridge
      // reports "not found" and we fall through silently.
      openInNativeApp(evt.run_dir + '/script.R', openScriptBtn, () =>
        openInNativeApp(evt.run_dir + '/script.do', openScriptBtn)
      );
    });
    actions.appendChild(openScriptBtn);

    const openFolderBtn = document.createElement('button');
    openFolderBtn.type = 'button';
    openFolderBtn.className = 'result-action';
    openFolderBtn.textContent = 'Show folder';
    openFolderBtn.title = 'Reveal the run directory in Finder.';
    openFolderBtn.addEventListener('click', () =>
      openInNativeApp(evt.run_dir, openFolderBtn)
    );
    actions.appendChild(openFolderBtn);

    panel.appendChild(actions);

    const note = document.createElement('div');
    note.className = 'result-note';
    note.textContent = evt.run_dir;
    panel.appendChild(note);
  }

  return panel;
}

async function openInNativeApp(path, btn, fallback) {
  /* Ask the Python bridge to hand the path to macOS `open`. The
   * bridge refuses anything outside Builder-managed directories, so
   * this can't be used to launch arbitrary files. */
  if (!window.pywebview || !window.pywebview.api) return;
  const originalText = btn && btn.textContent;
  if (btn) { btn.disabled = true; btn.textContent = 'Opening…'; }
  try {
    const result = await window.pywebview.api.open_path(path);
    if (!result || !result.ok) {
      if (fallback) { await fallback(); return; }
      // Transient inline error on the button itself.
      if (btn) btn.textContent = result && result.reason
        ? 'Error: ' + result.reason
        : 'Failed';
      setTimeout(() => {
        if (btn) btn.textContent = originalText;
      }, 3000);
    } else {
      // Don't reset the text immediately — the app is launching.
      // Restore shortly so the button doesn't look stuck.
      if (btn) {
        btn.textContent = 'Opened';
        setTimeout(() => { btn.textContent = originalText; }, 1500);
      }
    }
  } catch (e) {
    if (btn) btn.textContent = 'Failed';
    setTimeout(() => { if (btn) btn.textContent = originalText; }, 2000);
  } finally {
    if (btn) btn.disabled = false;
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

function buildPolicyCard(policy) {
  /* A pinned card at the top of the chat showing each dataset with
   * a depth-tier dropdown. Changing a dropdown persists immediately
   * to `.builder/policy.json` via the bridge. The researcher never
   * touches the JSON file or memorizes tier names — they pick from
   * English labels in a native <select>. */
  if (!policy) return null;
  const datasets = policy.datasets || [];
  if (datasets.length === 0) return null;

  const card = document.createElement('div');
  card.id = 'policy-card';
  card.className = 'policy-card';

  const header = document.createElement('div');
  header.className = 'policy-card-header';
  header.innerHTML =
    '<strong>Schema policy</strong> <span class="policy-card-sub">— what Claude sees about each dataset. Default is conservative; widen per-dataset if labels or summary counts aren\'t sensitive.</span>';
  card.appendChild(header);

  const list = document.createElement('div');
  list.className = 'policy-card-list';

  datasets.forEach((d) => {
    const row = document.createElement('div');
    row.className = 'policy-row';

    const name = document.createElement('span');
    name.className = 'policy-row-name';
    name.textContent = d.name;
    row.appendChild(name);

    const select = document.createElement('select');
    select.className = 'policy-row-select';
    select.dataset.dataset = d.name;
    DEPTH_TIERS.forEach((tier) => {
      const opt = document.createElement('option');
      opt.value = tier.value;
      opt.textContent = tier.label;
      if (tier.value === d.ceiling) opt.selected = true;
      select.appendChild(opt);
    });
    select.addEventListener('change', async () => {
      const depth = select.value;
      const prev = [...select.options].find((o) => o.defaultSelected);
      select.disabled = true;
      try {
        const result = await window.pywebview.api.set_dataset_policy(
          d.name, depth
        );
        if (!result || !result.ok) {
          // Revert on failure.
          if (prev) select.value = prev.value;
          // Inline error hint.
          const err = document.createElement('span');
          err.className = 'policy-row-err';
          err.textContent =
            ' ' + (result && result.reason ? result.reason : 'update failed');
          row.appendChild(err);
          setTimeout(() => err.remove(), 4000);
        } else {
          // Mark the new value as the default-selected one so a
          // future revert lands back here.
          [...select.options].forEach((o) => { o.defaultSelected = false; });
          select.options[select.selectedIndex].defaultSelected = true;
        }
      } catch (e) {
        if (prev) select.value = prev.value;
      } finally {
        select.disabled = false;
      }
    });
    row.appendChild(select);
    list.appendChild(row);
  });

  card.appendChild(list);
  return card;
}

// Signal to the Python side that we're ready to receive events. pywebview
// sets window.pywebview once its bridge is ready; until then we wait.
function whenReady(fn) {
  if (window.pywebview && window.pywebview.api) return fn();
  window.addEventListener('pywebviewready', fn, { once: true });
}

whenReady(async () => {
  // On startup, check whether the backend already has a cwd (launched
  // with an argv path) or needs one (land on the drop / choose-files
  // screen). ui_ready returns a synchronous response — no event yet.
  try {
    const state = await window.pywebview.api.ui_ready();
    if (state && state.state === 'ready') {
      showChat(state);
    } else {
      showLanding();
    }
  } catch (err) {
    console.error('ui_ready failed', err);
    showLanding();
  }
});
