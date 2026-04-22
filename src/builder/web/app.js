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
const stopBtn = document.getElementById('stop-btn');
const welcomeEl = document.getElementById('welcome');
const cwdEl = document.getElementById('cwd-display');

// Depth tiers — kept in sync with builder/policy.py::VALID_DEPTHS and
// with the get_schema tool help. Displayed labels are researcher-
// facing — plain English, not internal identifiers. "(default)" on
// the names_types row flags the app-wide default choice, so
// researchers know which option is the "unset" state.
const DEPTH_TIERS = [
  { value: 'names_only',                 label: 'Names only' },
  { value: 'names_types',                label: 'Names and types (default)' },
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
  // Policy is now a chip + popup in the composer, not a card at
  // the top of the message list. Re-render chip label + popup
  // contents from the fresh policy snapshot.
  updatePolicyChip(payload.policy);
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
  const rejected = files.length - accepted.length;
  if (accepted.length === 0) {
    setLandingError(
      'Drop .csv, .dta, or .rds files. Other types are ignored.'
    );
    return;
  }
  try {
    // Read serially with a progress message so large drops don't
    // look frozen. readAsDataURL loads the whole file into memory —
    // fine up to the 2 GB per-file cap, above which "Choose files…"
    // is the right path (see the landing fineprint).
    const payload = [];
    for (let i = 0; i < accepted.length; i++) {
      const file = accepted[i];
      const sizeMb = Math.round(file.size / (1024 * 1024));
      setLandingBusy(
        true,
        `Reading (${i + 1}/${accepted.length}) ${file.name}` +
          (sizeMb > 0 ? ` (${sizeMb} MB)…` : '…')
      );
      payload.push(await readFileAsBase64(file));
    }
    setLandingBusy(
      true,
      `Staging ${accepted.length} file${accepted.length === 1 ? '' : 's'}…`
    );
    const result = await window.pywebview.api.upload_files(payload);
    if (result && result.ok && rejected > 0) {
      // Will switch views — the note on rejected-types is just
      // a courtesy; no need to block.
      console.info(`${rejected} non-data file(s) ignored.`);
    }
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

// Turn lifecycle: the bridge's send_message is fire-and-forget (it
// queues the turn on the asyncio worker and returns immediately),
// so we can't tie the "is Claude still working" state to the await
// on that call. Instead we latch `turnInFlight` to true on submit
// and clear it when a terminal event arrives (turn_done /
// turn_error / auth_failure). That keeps the Send button disabled
// — and blocks Enter-to-send — while the prior turn is running,
// so a quick tester can't pipeline prompts that interleave in the
// transcript.

let turnInFlight = false;

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (turnInFlight) return;
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
    // Don't clear setSending here — the await resolves as soon as
    // the turn is QUEUED on the Python side, not when it finishes.
    // The turn_done / turn_error / auth_failure event handler
    // below is what flips the button back on.
  } catch (err) {
    appendError('send failed: ' + err);
    setSending(false);
  }
});

// Shift-Enter inserts a newline; plain Enter sends (but only when
// a turn isn't already in flight — avoids queuing multiple prompts
// by mashing Enter while Claude is thinking).
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (turnInFlight) return;
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
  turnInFlight = sending;
  // Toggle the Send / Stop icons rather than disabling the Send
  // button. During a turn, Stop replaces Send in the same spot so
  // the composer footprint doesn't reflow.
  if (sending) {
    sendBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');
  } else {
    stopBtn.classList.add('hidden');
    sendBtn.classList.remove('hidden');
  }
  input.setAttribute('aria-busy', sending ? 'true' : 'false');
}

// Stop button — asks the bridge to cancel the in-flight turn. The
// bridge cancels the asyncio task and tears down the SDK client so
// no half-finished request leaks into the next turn; a terminal
// "turn_error: cancelled" event arrives via the normal stream and
// clears the UI state through setSending(false).
if (stopBtn) {
  stopBtn.addEventListener('click', async () => {
    if (!window.pywebview || !window.pywebview.api) return;
    stopBtn.disabled = true;
    try {
      await window.pywebview.api.interrupt_turn();
    } catch (_) {
      // swallow — the turn_error event below will re-enable things
    } finally {
      stopBtn.disabled = false;
    }
  });
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
      // Terminal event: re-enable the composer. Token count could
      // surface in a tray later; not necessary for the chat-flow
      // guarantee this handler enforces.
      setSending(false);
      break;
    case 'auth_failure':
      appendError('Auth failure: ' + (evt.reason || 'unknown'));
      setSending(false);
      break;
    case 'turn_error':
      appendError(evt.message || 'unknown error');
      setSending(false);
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

  // Results from submit_script / expand_result carry a run_dir. Those
  // get a full result panel with the native R/Stata output and action
  // buttons ("Open output", "Open script in R/Stata", "Show folder").
  // The panel appears whether or not raw_stdout is populated — the
  // action buttons are useful even for scripts that produced no
  // console output, because the staged .R / .do file is still there
  // to re-open in RStudio / Stata.
  const hasRunDir = !!evt.run_dir;

  if (existingCard) {
    if (evt.is_error) {
      existingCard.classList.add('error');
    }
    const statusEl = existingCard.querySelector('.tool-status');
    if (statusEl) statusEl.textContent = evt.is_error ? 'error' : 'done';
    const body = existingCard.querySelector('.tool-body');

    if (hasRunDir) {
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
    // No matching call card (unusual) — render a standalone.
    if (hasRunDir) {
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
  /* Result panel for submit_script / expand_result events. Layout:
   *   ┌─ R / Stata output (always visible if non-empty) ─┐
   *   │  <pre>...native regression table...</pre>         │
   *   ├─ stderr (if present, yellow-tinted) ─────────────┤
   *   │  <pre>warnings...</pre>                           │
   *   ├─ Sanitized output (collapsed, toggle) ───────────┤
   *   │  <pre>{ ... clamped JSON ... }</pre>              │
   *   ├─ [Open output] [Open script] [Show folder] ─────┤
   *   └───────────────────────────────────────────────────┘
   * The stdout section is skipped entirely when raw output is
   * empty — otherwise we'd show a useless "(no output)" panel.
   * The sanitized section and action buttons are always present
   * for run_dir results so the researcher can re-open the script
   * even if it didn't print. */
  const panel = document.createElement('div');
  panel.className = 'result-panel' + (evt.is_error ? ' error' : '');

  const stdoutText = (evt.raw_stdout || '').trimEnd();
  if (stdoutText) {
    const stdoutSection = document.createElement('section');
    stdoutSection.className = 'result-stdout';
    const stdoutHeader = document.createElement('div');
    stdoutHeader.className = 'result-header';
    stdoutHeader.textContent = 'R / Stata output';
    stdoutSection.appendChild(stdoutHeader);
    const stdoutPre = document.createElement('pre');
    stdoutPre.textContent = stdoutText;
    stdoutSection.appendChild(stdoutPre);
    panel.appendChild(stdoutSection);
  }

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
    const actions = document.createElement('div');
    actions.className = 'result-actions';

    // Language-aware labels. ``evt.language`` is "R" or "Stata" when
    // the event came from submit_script / expand_result; falls back
    // to generic label otherwise.
    const lang = evt.language;  // "R" | "Stata" | undefined
    const scriptFile =
      lang === 'Stata' ? 'script.do' : 'script.R';
    const openInLabel =
      lang === 'Stata' ? 'Open in Stata'
        : lang === 'R' ? 'Open in R'
        : 'Open script';
    const openMode =
      lang === 'Stata' ? 'run_stata'
        : lang === 'R' ? 'run_r'
        : null;

    const openOutputBtn = document.createElement('button');
    openOutputBtn.type = 'button';
    openOutputBtn.className = 'result-action';
    openOutputBtn.textContent = 'Open output';
    openOutputBtn.title =
      'Open the full R/Stata stdout log in your default text editor.';
    openOutputBtn.addEventListener('click', () =>
      openInNativeApp(evt.run_dir + '/stdout.log', openOutputBtn, null)
    );
    actions.appendChild(openOutputBtn);

    const openScriptBtn = document.createElement('button');
    openScriptBtn.type = 'button';
    openScriptBtn.className = 'result-action';
    openScriptBtn.textContent = openInLabel;
    openScriptBtn.title =
      lang === 'Stata'
        ? 'Launch Stata with the script loaded. Press Cmd-D in the '
          + 'do-file editor to run it.'
        : lang === 'R'
        ? 'Launch RStudio with the script loaded. Press Cmd-Enter '
          + '(or Cmd-Shift-S) to run it.'
        : 'Open the R or Stata script in its default app.';
    openScriptBtn.addEventListener('click', () => {
      const primary = evt.run_dir + '/' + scriptFile;
      const fallback = evt.run_dir + '/' + (scriptFile === 'script.R' ? 'script.do' : 'script.R');
      openInNativeApp(primary, openScriptBtn, fallback, openMode);
    });
    actions.appendChild(openScriptBtn);

    const openFolderBtn = document.createElement('button');
    openFolderBtn.type = 'button';
    openFolderBtn.className = 'result-action';
    openFolderBtn.textContent = 'Show folder';
    openFolderBtn.title = 'Reveal the run directory in Finder.';
    openFolderBtn.addEventListener('click', () =>
      openInNativeApp(evt.run_dir, openFolderBtn, null)
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

async function openInNativeApp(path, btn, fallback, mode) {
  /* Ask the Python bridge to hand the path to macOS `open`. ``mode``
   * selects a Stata/R-specific launch (open -a StataMP / RStudio).
   * ``fallback`` is a path to try if the primary fails. */
  if (!window.pywebview || !window.pywebview.api) return;
  const originalText = btn && btn.textContent;
  if (btn) { btn.disabled = true; btn.textContent = 'Opening…'; }
  try {
    const result = await window.pywebview.api.open_path(path, mode || null);
    if (!result || !result.ok) {
      if (fallback) {
        const f = await window.pywebview.api.open_path(fallback, mode || null);
        if (f && f.ok) {
          if (btn) {
            btn.textContent = 'Opened';
            setTimeout(() => { btn.textContent = originalText; }, 1500);
          }
          return;
        }
      }
      if (btn) btn.textContent = result && result.reason
        ? 'Error: ' + result.reason
        : 'Failed';
      setTimeout(() => {
        if (btn) btn.textContent = originalText;
      }, 3000);
    } else {
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

// ----- policy chip + popup (next to composer) ----------------------------

const policyChip = document.getElementById('policy-chip');
const policyChipLabel = document.getElementById('policy-chip-label');
const policyPopup = document.getElementById('policy-popup');
let policyPopupBuiltFor = null;  // cached copy so we don't rebuild needlessly

function updatePolicyChip(policy) {
  /* Refresh the compact chip label + the per-dataset dropdowns
   * inside the popup. Called on session start and after any policy
   * mutation. */
  if (!policy || !policy.datasets || policy.datasets.length === 0) {
    policyChip.classList.add('hidden');
    return;
  }
  policyChip.classList.remove('hidden');
  policyChipLabel.textContent = compactPolicyLabel(policy);
  policyPopupBuiltFor = policy;
  policyPopup.innerHTML = '';
  policyPopup.appendChild(buildPolicyPopup(policy));
}

function compactPolicyLabel(policy) {
  // Keep the chip short so it fits next to Send. Details live in
  // the popup the chip opens. Show a customization count only when
  // it would be useful information.
  const customized = policy.datasets.filter((d) => d.explicit).length;
  if (customized === 0) return 'Policy';
  return `Policy · ${customized} custom`;
}

function buildPolicyPopup(policy) {
  const wrapper = document.createElement('div');
  const header = document.createElement('div');
  header.className = 'policy-popup-header';
  header.innerHTML =
    '<strong>Schema policy</strong><br>' +
    'What Claude sees about each dataset. Default is conservative; ' +
    'widen per-dataset if labels or counts aren\'t sensitive.';
  wrapper.appendChild(header);

  const list = document.createElement('div');
  list.className = 'policy-card-list';

  policy.datasets.forEach((d) => {
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
          if (prev) select.value = prev.value;
          const err = document.createElement('span');
          err.className = 'policy-row-err';
          err.textContent = ' ' + (result && result.reason ? result.reason : 'failed');
          row.appendChild(err);
          setTimeout(() => err.remove(), 4000);
        } else {
          [...select.options].forEach((o) => { o.defaultSelected = false; });
          select.options[select.selectedIndex].defaultSelected = true;
          // Re-render the compact chip label to reflect the new
          // customized count.
          if (result.policy) updatePolicyChip(result.policy);
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

  wrapper.appendChild(list);
  return wrapper;
}

// Chip click toggles the popup; click outside closes it.
policyChip.addEventListener('click', (e) => {
  e.stopPropagation();
  const isOpen = !policyPopup.classList.contains('hidden');
  policyPopup.classList.toggle('hidden');
  policyChip.classList.toggle('open', !isOpen);
});
document.addEventListener('click', (e) => {
  if (policyPopup.classList.contains('hidden')) return;
  if (policyPopup.contains(e.target) || policyChip.contains(e.target)) return;
  policyPopup.classList.add('hidden');
  policyChip.classList.remove('open');
});

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
