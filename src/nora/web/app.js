/* Nora web UI — frontend JS.
 *
 * Communicates with the Python backend via pywebview's bridge:
 *   window.pywebview.api.<method>(...)    — JS → Python
 *   Python calls `window.nora_event({type, ...payload})` via
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
const cwdEl = document.getElementById('cwd-display');
const contextChip = document.getElementById('context-chip');

// Context-window ceiling for the chip's ratio display. Updated
// whenever the researcher picks a model (see updateModelChip) —
// Sonnet 4.6 defaults to 1M, Opus 4.7 and Haiku 4.5 to 200k. The
// starting 1M matches the default model (Sonnet).
const DEFAULT_CONTEXT_WINDOW = 1_000_000;
let contextWindow = DEFAULT_CONTEXT_WINDOW;

// Persisted model choice — survives restarts. Applied on boot after
// the bridge is ready (loadModels runs `set_model` if the stored
// value differs from the backend default).
const MODEL_STORAGE_KEY = 'nora.model';

// ----- theme toggle -------------------------------------------------------
// Temporary light/dark override. When the user hasn't clicked the
// toggle, the CSS `prefers-color-scheme` media query picks the theme
// from the OS. Once they click, we set `data-theme` on <html> and
// persist to localStorage so the choice survives restarts. Clearing
// the entry (via the same toggle or dev tools) returns to OS
// follow-mode.
const THEME_STORAGE_KEY = 'nora.theme';
const themeToggleBtn = document.getElementById('theme-toggle');
const themeToggleIcon = document.getElementById('theme-toggle-icon');

function currentTheme() {
  // Effective theme (light | dark): whatever data-theme says, or
  // the OS preference if no override is set.
  const forced = document.documentElement.getAttribute('data-theme');
  if (forced === 'light' || forced === 'dark') return forced;
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark' : 'light';
}

function renderThemeIcon() {
  if (!themeToggleIcon) return;
  // Glyph shows the theme you'll jump TO on click, matching the
  // affordance researchers expect from every other theme toggle.
  themeToggleIcon.textContent = currentTheme() === 'dark' ? '☀' : '☾';
}

function applyStoredTheme() {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') {
      document.documentElement.setAttribute('data-theme', stored);
    }
  } catch (_) { /* localStorage blocked — fall back to OS default */ }
  renderThemeIcon();
}

if (themeToggleBtn) {
  themeToggleBtn.addEventListener('click', () => {
    const next = currentTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem(THEME_STORAGE_KEY, next); } catch (_) {}
    renderThemeIcon();
  });
}

// OS theme change while Nora is open: keep the icon in sync
// (only matters when no explicit override is set).
window.matchMedia('(prefers-color-scheme: dark)').addEventListener(
  'change', renderThemeIcon
);

applyStoredTheme();

// Depth tiers — kept in sync with nora/policy.py::VALID_DEPTHS and
// with the get_schema tool help. Labels are researcher-facing plain
// English using an additive "+ X" convention: each tier adds what
// the previous tier sees, so the ladder reads naturally top to
// bottom. No "(default)" marker here — the current tier is implied
// by the <select>'s own selected-state, and the backend default
// (DEFAULT_MAX_DEPTH in policy.py) decides which one that is.
const DEPTH_TIERS = [
  { value: 'names_only',                 label: 'Variable names only' },
  { value: 'names_types',                label: '+ types' },
  { value: 'names_types_labels',         label: '+ labels / value labels' },
  { value: 'names_types_labels_summary', label: '+ NA count / distinct count' },
];

// ----- view routing ------------------------------------------------------

function showLanding() {
  landingEl.classList.remove('hidden');
  chatEl.classList.add('hidden');
  // Reset landing-side UI so re-entering from "New session" feels
  // fresh: re-enable both buttons, clear any leftover status text,
  // drop the drag-over highlight. Without this, a researcher who
  // cancelled a previous file-picker or came back from chat sees
  // greyed-out buttons and thinks the page is frozen.
  setLandingBusy(false, '');
  dropZone.classList.remove('dragover');
  // Show past sessions underneath the upload area so a researcher
  // can jump straight back into a prior working directory.
  loadLandingSessions();
}

async function loadLandingSessions() {
  const container = document.getElementById('landing-sessions');
  const listEl = document.getElementById('landing-sessions-list');
  if (!container || !listEl) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.list_sessions !== 'function') return;
  try {
    const res = await window.pywebview.api.list_sessions();
    if (!res || !res.ok) return;
    const sessions = res.sessions || [];
    if (sessions.length === 0) {
      container.classList.add('hidden');
      return;
    }
    container.classList.remove('hidden');
    listEl.innerHTML = '';
    // Cap to the 5 most recent so the landing page doesn't become
    // a wall of text. Full list is always available in the sidebar
    // once the researcher is in a session.
    sessions.slice(0, 5).forEach((s) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'landing-session';
      btn.title = s.path;

      const when = document.createElement('div');
      when.className = 'landing-session-when';
      when.textContent = formatSessionWhen(s.timestamp);
      btn.appendChild(when);

      const meta = document.createElement('div');
      meta.className = 'landing-session-meta';
      const dsText = s.datasets.length
        ? s.datasets.join(', ')
        : '(no data files)';
      const sizeText = typeof s.size === 'number' ? formatBytes(s.size) : '';
      meta.textContent = sizeText ? `${sizeText} · ${dsText}` : dsText;
      btn.appendChild(meta);

      btn.addEventListener('click', () => switchSession(s.path, false));
      listEl.appendChild(btn);
    });
  } catch (_) { /* silent — landing shouldn't block on history */ }
}

function showChat(payload) {
  /* Reveal the chat view and populate it from a ready payload.
   * Called both on initial startup (when the backend already has a
   * cwd) and after a session switch, so it has to reset any prior
   * transcript rather than just updating pieces in place. */
  landingEl.classList.add('hidden');
  chatEl.classList.remove('hidden');

  // Reset the transcript to a fresh welcome placeholder.
  // replayHistory() below will clear and repopulate this if the
  // target session has persisted events on disk.
  messagesEl.innerHTML = '';
  const welcomeMsg = document.createElement('div');
  // .welcome-greeting is a permanent class (not toggled with
  // welcome-only) so the horizontal centering — full chat-area
  // width with body justify-content:center — applies even AFTER
  // the first message arrives. Without this, removing welcome-only
  // dropped the welcome back into the 960px column cap, which
  // sits left-of-center on a wide window because the sidebar
  // eats space on the left side.
  welcomeMsg.className = 'message system welcome-greeting';
  const welcomeBody = document.createElement('div');
  welcomeBody.className = 'message-body';
  welcomeBody.id = 'welcome';
  welcomeBody.textContent = payload.greeting || 'Ready.';
  welcomeMsg.appendChild(welcomeBody);
  messagesEl.appendChild(welcomeMsg);
  setWelcomeOnlyMode(true);

  // Topbar shows a friendly session title (dataset name or a
  // timestamped "Session ..." label), not the raw path. Full path
  // is still available on hover via the `title` attribute.
  cwdEl.textContent = payload.session_title || formatCwd(payload.cwd || '');
  cwdEl.title = payload.cwd || '';

  updatePolicyChip(payload.policy);
  loadSessions();
  loadModels();
  // New conversation (or resumed one) so the context-usage chip
  // stays hidden until the next turn_done provides real numbers.
  if (contextChip) contextChip.classList.add('hidden');

  replayHistory();
  rotatePlaceholder();
  input.focus();
}

// Set while we're replaying a persisted chat log so appendAssistant
// skips the typewriter animation — past messages should appear all
// at once, not trickle in for several seconds per bubble.
let replayMode = false;

async function replayHistory() {
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.get_chat_history !== 'function') return;
  try {
    const res = await window.pywebview.api.get_chat_history();
    if (!res || !res.ok) return;
    const events = res.events || [];
    if (events.length === 0) return;
    // Drop the default welcome "system" line so replayed history
    // starts the transcript instead of below the greeting.
    messagesEl.innerHTML = '';
    setWelcomeOnlyMode(false);
    replayMode = true;
    try {
      events.forEach((evt) => replayEvent(evt));
    } finally {
      replayMode = false;
    }
    scrollToBottom();
  } catch (err) {
    console.warn('get_chat_history failed', err);
  }
}

function replayEvent(evt) {
  /* Shared handler for replayed events. Live events go through
   * window.nora_event; replay feeds the same records back
   * through the same append* helpers so rendering is identical.
   * Named deliberately to avoid shadowing window.dispatchEvent,
   * which is a DOM method and was my first attempt. */
  switch (evt.type) {
    case 'user_message':
      appendUser(evt.text || '');
      break;
    case 'assistant_text':
      appendAssistant(evt.text || '');
      break;
    case 'assistant_thinking':
      appendThinking(evt.text || '');
      break;
    case 'tool_call':
      appendToolCall(evt);
      break;
    case 'tool_result':
      appendToolResult(evt);
      break;
  }
}

function formatCwd(raw) {
  /* Abbreviate the working-directory path for display in the topbar
   * chip. The full path —
   * ``/Users/bb/.nora-sessions/20260422T160059Z_f13630f4`` —
   * overflows the chip and the timestamp at the tail is the part
   * the researcher actually wants to see (which session). We
   * collapse ``/Users/<user>`` to ``~`` and cap at 40 chars with a
   * left-side ellipsis, preserving the tail. Full path remains
   * available via the ``title`` tooltip.
   */
  if (!raw) return '';
  let p = raw.replace(/^\/Users\/[^/]+/, '~');
  const MAX = 40;
  if (p.length <= MAX) return p;
  return '…' + p.slice(-(MAX - 1));
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

// Staged images for the next message. Populated by drop / paste,
// cleared on send. Each entry: { data: base64String, mime: string,
// url: objectURL (for thumbnail preview) }.
const stagedImages = [];
const attachmentsEl = document.getElementById('compose-attachments');
const ALLOWED_IMAGE_MIMES = new Set([
  'image/png', 'image/jpeg', 'image/webp', 'image/gif',
]);
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;  // 5 MB per image (Anthropic limit ballpark)

function renderAttachments() {
  if (!attachmentsEl) return;
  attachmentsEl.innerHTML = '';
  if (stagedImages.length === 0 && stagedDataNotices.length === 0) {
    attachmentsEl.classList.add('hidden');
    return;
  }
  attachmentsEl.classList.remove('hidden');
  stagedImages.forEach((img, idx) => {
    const wrap = document.createElement('div');
    wrap.className = 'compose-attachment';
    const thumb = document.createElement('img');
    thumb.src = img.url;
    thumb.alt = 'Staged image ' + (idx + 1);
    wrap.appendChild(thumb);
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'compose-attachment-remove';
    rm.setAttribute('aria-label', 'Remove');
    rm.textContent = '×';
    rm.addEventListener('click', () => {
      URL.revokeObjectURL(img.url);
      stagedImages.splice(idx, 1);
      renderAttachments();
    });
    wrap.appendChild(rm);
    attachmentsEl.appendChild(wrap);
  });
  // Named-chip notices for data/script files the researcher just
  // added. These have no payload to send (the file is already on
  // disk); they're just a visual "yes that landed" receipt.
  stagedDataNotices.forEach((name, idx) => {
    const chip = document.createElement('div');
    chip.className = 'compose-attachment compose-attachment-file';
    chip.title = name + ' — copied into the session';
    const label = document.createElement('span');
    label.className = 'compose-attachment-filename';
    label.textContent = name;
    chip.appendChild(label);
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'compose-attachment-remove';
    rm.setAttribute('aria-label', 'Dismiss');
    rm.textContent = '×';
    rm.addEventListener('click', () => {
      stagedDataNotices.splice(idx, 1);
      renderAttachments();
    });
    chip.appendChild(rm);
    attachmentsEl.appendChild(chip);
  });
}

async function stageImageFile(file) {
  if (!ALLOWED_IMAGE_MIMES.has(file.type)) {
    appendError('Only PNG, JPEG, WebP, and GIF images are supported.');
    return;
  }
  if (file.size > MAX_IMAGE_BYTES) {
    appendError('Image too large — max 5 MB per file.');
    return;
  }
  const data = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const url = reader.result;
      const comma = url.indexOf(',');
      resolve(comma >= 0 ? url.slice(comma + 1) : url);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  stagedImages.push({
    data,
    mime: file.type,
    url: URL.createObjectURL(file),
  });
  renderAttachments();
}

// Extensions we accept on composer drop/paste alongside images.
// Same set the + button's picker accepts — data files, R/Stata
// scripts, Stata graphs, logs, and R Markdown. All copied into
// the session cwd so Claude can reference them.
const COMPOSER_DATA_EXTS = new Set([
  'csv', 'dta', 'rds',
  'do', 'r',
  'gph',
  'log', 'smcl',
  'rmd',
]);

function fileExt(file) {
  const parts = (file.name || '').split('.');
  if (parts.length < 2) return '';
  return parts.pop().toLowerCase();
}

function acceptedByComposer(file) {
  if (ALLOWED_IMAGE_MIMES.has(file.type)) return true;
  return COMPOSER_DATA_EXTS.has(fileExt(file));
}

// Stage a non-image data/script file by shipping it to the backend,
// which copies it into the session cwd. Shows a named chip in the
// attachment bar as confirmation. Errors go into the chat transcript.
async function stageDataFile(file) {
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.add_files_from_blobs !== 'function') {
    appendError('Restart Nora to drop files into the chat.');
    return;
  }
  const data = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const url = reader.result;
      const comma = url.indexOf(',');
      resolve(comma >= 0 ? url.slice(comma + 1) : url);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  try {
    const res = await window.pywebview.api.add_files_from_blobs([
      { name: file.name, content: data, mime: file.type || '' },
    ]);
    if (!res || !res.ok) {
      appendError(friendlyAddFilesError(res && res.reason ? res.reason : 'unknown'));
      return;
    }
    if (res.added && res.added.length > 0) {
      stagedDataNotices.push(res.added[0]);
      renderAttachments();
    }
    if (res.skipped && res.skipped.length > 0) {
      appendError('Skipped: ' + res.skipped.join(', '));
    }
    if (res.policy) updatePolicyChip(res.policy);
    if (res.session_title && cwdEl) cwdEl.textContent = res.session_title;
    if (typeof loadSessions === 'function') loadSessions();
  } catch (err) {
    appendError(friendlyAddFilesError(err && err.message ? err.message : String(err)));
  }
}

// Named-chip notices for data/script files the researcher just
// added via composer drop/paste. Pure visual confirmation — the
// file is already on disk. Cleared on send or when the researcher
// removes the chip.
const stagedDataNotices = [];

// Drop handling on the compose form. Accepts images (staged as
// vision attachments) and data/script files (.csv, .dta, .rds,
// .do, .r, .log, .smcl, .gph, .rmd — copied into the session cwd).
// Landing's drop-zone handles fresh-session files on the landing
// screen; this handler is for mid-session additions.
if (form) {
  ['dragenter', 'dragover'].forEach((name) => {
    form.addEventListener(name, (e) => {
      if (!e.dataTransfer) return;
      // Check items (not files) on dragover — files is empty during
      // drag on most browsers for security reasons. items gives us
      // the MIME type but not the filename, so we only highlight
      // when at least one clearly-usable file is being dragged.
      const items = Array.from(e.dataTransfer.items || []);
      const hasUsable = items.some((it) =>
        it.kind === 'file' && ALLOWED_IMAGE_MIMES.has(it.type)
      );
      if (!hasUsable && items.length === 0) return;
      // Even if we can't confirm a usable file (items lacks ext
      // info for non-image drops), allow the drop — we'll filter
      // on the drop event where filenames are available.
      e.preventDefault();
      form.classList.add('drop-target');
    });
  });
  form.addEventListener('dragleave', (e) => {
    if (e.target === form) form.classList.remove('drop-target');
  });
  form.addEventListener('drop', async (e) => {
    const allFiles = Array.from(e.dataTransfer?.files || []);
    const usable = allFiles.filter(acceptedByComposer);
    if (usable.length === 0) return;
    e.preventDefault();
    form.classList.remove('drop-target');
    const skipped = allFiles.filter((f) => !acceptedByComposer(f));
    if (skipped.length > 0) {
      appendError(
        'Skipped: ' + skipped.map((f) => f.name).join(', ') +
        ' — only images and data/script files (.csv, .dta, .rds, .do, .r, .log, .smcl, .gph, .rmd) can be dropped here.'
      );
    }
    for (const file of usable) {
      if (ALLOWED_IMAGE_MIMES.has(file.type)) {
        await stageImageFile(file);
      } else {
        await stageDataFile(file);
      }
    }
    input.focus();
  });
}

// Cmd-V / Ctrl-V pasting into the composer. Images stage as vision
// attachments; data/script files (if pasted from Finder) land in
// the session cwd just like a drop.
if (input) {
  input.addEventListener('paste', async (e) => {
    const items = Array.from(e.clipboardData?.items || []);
    const fileItems = items.filter((it) => it.kind === 'file');
    if (fileItems.length === 0) return;
    const usable = [];
    for (const it of fileItems) {
      const f = it.getAsFile();
      if (f && acceptedByComposer(f)) usable.push(f);
    }
    if (usable.length === 0) return;
    e.preventDefault();
    for (const f of usable) {
      if (ALLOWED_IMAGE_MIMES.has(f.type)) {
        await stageImageFile(f);
      } else {
        await stageDataFile(f);
      }
    }
  });
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (turnInFlight) return;
  const text = input.value.trim();
  const images = stagedImages.slice();  // snapshot
  if (!text && images.length === 0) return;
  if (!window.pywebview || !window.pywebview.api) {
    appendSystem('backend not ready yet; try again');
    return;
  }
  appendUser(text || '(image only)');
  input.value = '';
  autosize();
  rotatePlaceholder();
  // Clear staged images from the UI immediately — the snapshot
  // carries them to the backend. Data-file notices are receipts
  // only, no payload to send; clear them at the same point so the
  // composer returns to a clean state after each turn.
  stagedImages.forEach((img) => URL.revokeObjectURL(img.url));
  stagedImages.length = 0;
  stagedDataNotices.length = 0;
  renderAttachments();
  setSending(true);
  try {
    // If images are attached, use the richer send method. The
    // simpler string send stays as the fast path for text-only.
    if (images.length > 0 && typeof window.pywebview.api.send_message_with_images === 'function') {
      const payload = images.map((img) => ({ data: img.data, mime: img.mime }));
      await window.pywebview.api.send_message_with_images(text, payload);
    } else if (images.length > 0) {
      appendError('Restart Nora to send images.');
      setSending(false);
      return;
    } else {
      await window.pywebview.api.send_message(text);
    }
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

// Rotating placeholder — short, slightly silly, changes each time
// the input goes empty (initial load, after send, after clearing).
const PLACEHOLDERS = [
  'Nam nam nam data…',
  'I grant your wishes',
  'Hello there, I am Nora',
  'What shall we regress today',
  'Coefficients on tap',
  'Feed me a question',
  'Standard errors, on the house',
  'Poke the dataset',
  'Tell me where it hurts (in the data)',
  'Ready when you are',
  'A t-test? A table? Surprise me',
  'Pun buffer: loaded',
  'Ask away, researcher',
];

function rotatePlaceholder() {
  if (!input) return;
  const next = PLACEHOLDERS[Math.floor(Math.random() * PLACEHOLDERS.length)];
  input.setAttribute('placeholder', next);
}
rotatePlaceholder();

function setSending(sending) {
  turnInFlight = sending;
  // Toggle the Send / Stop icons rather than disabling the Send
  // button. During a turn, Stop replaces Send in the same spot so
  // the composer footprint doesn't reflow.
  if (sending) {
    sendBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');
    showLoadingIndicator();
  } else {
    stopBtn.classList.add('hidden');
    sendBtn.classList.remove('hidden');
    hideLoadingIndicator();
  }
  input.setAttribute('aria-busy', sending ? 'true' : 'false');
}

// Rotation of vague, slightly silly labels for the loading indicator.
// Picked at random each time the indicator shows; keeps a long wait
// from feeling monotonous. Kept short so the line doesn't reflow on
// narrow windows. Explicit "doing X to Y" labels are avoided — the
// researcher doesn't need to know whether we're peeking at a schema
// or running a script; they just need to know we're still at it.
const LOADING_LABELS = [
  'facticulating',
  'triangulating',
  'cogitating',
  'ruminating',
  'percolating',
  'synthesizing',
  'noodling',
  'pondering',
  'mulling',
  'musing',
  'deliberating',
  'contemplating',
  'chewing on it',
  'connecting dots',
  'chasing threads',
  'conjuring',
  'marinating',
  'consulting the oracle',
  'squinting at it',
  'untangling',
];

function showLoadingIndicator() {
  /* Append a small animated yarn-ball element at the bottom of the
   * transcript so the researcher has something to watch while Claude
   * is working. Removed on any terminal event (turn_done /
   * turn_error / auth_failure). Idempotent — multiple calls in a
   * row don't stack extra spinners. */
  if (document.getElementById('loading-indicator')) return;
  const label = LOADING_LABELS[
    Math.floor(Math.random() * LOADING_LABELS.length)
  ];
  const el = document.createElement('div');
  el.id = 'loading-indicator';
  el.className = 'loading-indicator';
  el.setAttribute('aria-label', 'Claude is ' + label);
  // Illustrated Lottie animation: a cat whose paw pushes a yarn
  // ball. Replaces the earlier CSS-only two-paws-and-a-ball stack —
  // more character, honest "retro-illustrated" feel, no hand-rolled
  // keyframes to maintain. The JSON is bundled at
  // src/builder/web/cat-loading.json; the player is the locally-
  // bundled lottie-player web component (see index.html). background
  // transparent so it themes cleanly light / dark.
  el.innerHTML =
    '<lottie-player class="cat-loading" src="cat-loading.json" ' +
    'background="transparent" speed="0.6" autoplay loop ' +
    'aria-hidden="true"></lottie-player>' +
    '<span class="loading-text">' + label + '</span>';
  messagesEl.appendChild(el);
  scrollToBottom();
}

function hideLoadingIndicator() {
  const el = document.getElementById('loading-indicator');
  if (el) el.remove();
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

window.nora_event = function (evt) {
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
      // Terminal event: re-enable the composer and update the
      // context-usage chip.
      //
      // The chip tracks the *prompt* side only:
      //   input_tokens + cache_read + cache_creation
      // i.e. what the model actually loaded into its window this
      // turn. ``output_tokens`` is deliberately excluded — Claude's
      // fresh response is not in the window on THIS turn; it gets
      // folded into input on the NEXT turn via cache_creation /
      // input_tokens. Including it here would double-count across
      // turns and make the chip bounce turn-to-turn based on how
      // long the response happened to be.
      setSending(false);
      const prompt =
        (evt.input_tokens || 0) +
        (evt.cache_read_input_tokens || 0) +
        (evt.cache_creation_input_tokens || 0);
      updateContextChip(prompt);
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
      updatePolicyChip(evt.policy);
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
  // Replay path: skip the animation entirely. Past messages should
  // land instantly so loading a session doesn't take N seconds per
  // assistant bubble.
  if (replayMode) {
    append('assistant', text || '', /*markdown=*/ true);
    return;
  }
  const textSafe = text || '';
  if (textSafe.length > 900) {
    append('assistant', textSafe, /*markdown=*/ true);
    return;
  }
  // The SDK hands us complete text blocks per turn, not token-by-token
  // deltas, so real streaming isn't available at this layer. To give
  // the conversation a "typing" feel anyway, we drop in the bubble
  // immediately and animate the text into it at a visible-but-snappy
  // rate, then swap to rendered markdown once the animation finishes.
  // Errors / tool calls arriving mid-animation force an instant
  // finish so the transcript order stays honest.
  finalizeActiveTypewriter();
  setWelcomeOnlyMode(false);
  const wrapper = document.createElement('div');
  wrapper.className = 'message assistant';
  const body = document.createElement('div');
  body.className = 'message-body typing';
  wrapper.appendChild(body);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
  runTypewriter(body, textSafe, () => {
    // Swap from plain-text animation to rendered markdown. Clearing
    // the `typing` class drops the caret and flips white-space back
    // to normal so paragraphs/lists/tables lay out correctly.
    body.classList.remove('typing');
    if (textSafe && window.NoraMarkdown) {
      body.innerHTML = window.NoraMarkdown.render(textSafe);
    } else {
      body.textContent = textSafe;
    }
    scrollToBottom();
  });
}

// The typewriter currently animating, if any. Tracked globally so
// any other UI event (new assistant block, tool_call, tool_result,
// turn_done) can force it to finish immediately — we never want a
// visual animation to outlive the event that follows it.
let activeTypewriter = null;

function finalizeActiveTypewriter() {
  if (activeTypewriter) activeTypewriter.finalize();
}

function runTypewriter(bodyEl, fullText, onComplete) {
  const len = fullText.length;
  if (len === 0) {
    onComplete();
    return;
  }
  // Adaptive speed: short messages land in ~500 ms; longer ones
  // cap at ~2.5 s so the "typing" cue reads clearly without making
  // the researcher wait through a crawl. Per-char pacing is ~12 ms
  // (~80 chars/sec) — slow enough to feel like typing, fast enough
  // that a full paragraph doesn't turn into a coffee break.
  const targetMs = Math.min(2500, Math.max(500, len * 12));
  const charsPerMs = len / targetMs;
  let typed = 0;
  let lastTime = performance.now();
  let rafId = 0;
  let finalized = false;

  const tw = {
    finalize() {
      if (finalized) return;
      finalized = true;
      cancelAnimationFrame(rafId);
      if (activeTypewriter === tw) activeTypewriter = null;
      onComplete();
    },
  };
  activeTypewriter = tw;

  function frame(now) {
    if (finalized) return;
    const dt = now - lastTime;
    lastTime = now;
    typed = Math.min(len, typed + Math.max(1, Math.ceil(charsPerMs * dt)));
    bodyEl.textContent = fullText.slice(0, typed);
    scrollToBottom();
    if (typed < len) {
      rafId = requestAnimationFrame(frame);
    } else {
      tw.finalize();
    }
  }
  rafId = requestAnimationFrame(frame);
}

function appendThinking(text) {
  /* Claude's reasoning trace. Rendered as a collapsible card so the
   * researcher can see WHAT Claude was thinking without the trace
   * dominating the transcript. Starts collapsed; clicking the
   * header expands. Mirrors the submit_script card shape so the
   * interaction model is consistent. */
  finalizeActiveTypewriter();
  setWelcomeOnlyMode(false);
  const card = document.createElement('div');
  card.className = 'thinking-card collapsed';

  const header = document.createElement('div');
  header.className = 'thinking-header';
  const arrow = document.createElement('span');
  arrow.className = 'thinking-arrow';
  arrow.textContent = '▼';
  const label = document.createElement('span');
  label.className = 'thinking-label';
  label.textContent = 'Thinking';
  header.appendChild(arrow);
  header.appendChild(label);

  const body = document.createElement('div');
  body.className = 'thinking-body';
  body.textContent = text;

  header.addEventListener('click', () => card.classList.toggle('collapsed'));

  card.appendChild(header);
  card.appendChild(body);
  messagesEl.appendChild(card);
  scrollToBottom();
}

function appendSystem(text) {
  append('system', text);
}

function appendError(text) {
  append('error', text);
}

function append(kind, text, markdown) {
  // User / system / error messages appear instantly. Make sure any
  // typewriter from the previous turn lands first, so a fresh user
  // bubble doesn't appear above a still-animating assistant bubble.
  setWelcomeOnlyMode(false);
  finalizeActiveTypewriter();
  const wrapper = document.createElement('div');
  wrapper.className = 'message ' + kind;
  const body = document.createElement('div');
  body.className = 'message-body';
  if (markdown && window.NoraMarkdown) {
    body.innerHTML = window.NoraMarkdown.render(text);
  } else {
    body.textContent = text;
  }
  wrapper.appendChild(body);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
}

function appendToolCall(evt) {
  // Only ``submit_script`` renders a card. ``get_schema``,
  // ``request_data``, ``expand_result``, ``list_results`` all happen
  // silently — they're plumbing, not results the researcher reads.
  // Claude summarizes whatever matters from them in the chat text
  // that follows.
  finalizeActiveTypewriter();
  setWelcomeOnlyMode(false);
  const shortName = shortenToolName(evt.name);
  if (shortName !== 'submit_script') return;

  const card = document.createElement('div');
  card.dataset.callId = evt.call_id;
  card.className = 'tool-card';
  const isSubmitScript = true;

  const header = document.createElement('div');
  header.className = 'tool-header';
  const arrow = document.createElement('span');
  arrow.className = 'tool-arrow';
  arrow.textContent = '▼';
  const title = document.createElement('span');
  title.innerHTML =
    '<span class="tool-name">' + shortName + '</span>' +
    ' <span class="tool-status">running…</span>';
  header.appendChild(arrow);
  header.appendChild(title);

  const body = document.createElement('div');
  body.className = 'tool-body';

  if (isSubmitScript) {
    // Render the language + code + label prominently, not as JSON
    // stringification. Researcher sees the actual script.
    const input = evt.input || {};
    const langText = (input.language || '').toString().toUpperCase();
    if (input.label) {
      const label = document.createElement('div');
      label.className = 'tool-label';
      label.textContent = input.label;
      body.appendChild(label);
    }
    const pre = document.createElement('pre');
    pre.className = 'tool-code lang-' + (input.language || 'text').toLowerCase();
    // Small language badge inside the code block so the researcher
    // knows whether this is R or Stata at a glance.
    const badge = document.createElement('span');
    badge.className = 'tool-lang-badge';
    badge.textContent = langText || 'script';
    pre.appendChild(badge);
    const codeEl = document.createElement('code');
    codeEl.textContent = input.code || '';
    pre.appendChild(codeEl);
    body.appendChild(pre);
    if (input.source_dataset) {
      const src = document.createElement('div');
      src.className = 'tool-source';
      src.textContent = 'source: ' + input.source_dataset;
      body.appendChild(src);
    }
  } else {
    // Other tools — show the input as JSON (it's short and
    // structural, so stringification is fine).
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(evt.input, null, 2);
    body.appendChild(pre);
  }

  header.addEventListener('click', () => card.classList.toggle('collapsed'));

  card.appendChild(header);
  card.appendChild(body);
  messagesEl.appendChild(card);
  scrollToBottom();
}

function appendToolResult(evt) {
  setWelcomeOnlyMode(false);
  // Non-submit_script tool calls don't create cards in
  // appendToolCall, so there's nothing to update here — silent pass.
  const existingCard = [...messagesEl.querySelectorAll('.tool-card')]
    .find((c) => c.dataset.callId === evt.call_id);
  if (!existingCard) return;

  if (evt.is_error) existingCard.classList.add('error');
  const statusEl = existingCard.querySelector('.tool-status');
  if (statusEl) statusEl.textContent = evt.is_error ? 'error' : 'done';
  const body = existingCard.querySelector('.tool-body');

  // submit_script only: show the native R/Stata result (post-
  // preamble) inline, plus the Open-in / Show-folder buttons.
  // Errors are not surfaced on the card — Claude's chat reply
  // explains what went wrong. The sanitized payload is not shown;
  // researchers who want it can ask Claude.
  renderScriptResultInline(body, evt);
  scrollToBottom();
}

// Marker emitted by the Stata preamble in executor.py. Everything
// above this line in stdout is plumbing (adopath, cd, comments the
// researcher didn't write); everything below is the researcher's
// actual script output — the regression table, summary stats, etc.
// R has no equivalent marker because ``source(nora.R)`` runs
// silently, so R stdout is already clean.
const STATA_PREAMBLE_MARKER =
  'Nora preamble above; researcher code below';

function stripPreamble(stdout, _language) {
  /* Split on the Stata preamble marker and return everything after
   * it. The ``_language`` argument is kept for signature stability,
   * but the decision is marker-based, not language-based: error
   * paths in tools.py (execution_failed / rejected_by_sanitizer)
   * don't set ``_language``, so gating on language==='Stata' would
   * leak the preamble on every failed run. Since the marker is only
   * injected into Stata stdout, R output is unaffected either way. */
  if (!stdout) return '';
  const idx = stdout.indexOf(STATA_PREAMBLE_MARKER);
  if (idx < 0) return stdout;
  // Marker sits inside a `.*! ----- ... -----` comment line. Jump
  // past the end of that line so the caller sees clean output from
  // the first researcher command onwards.
  const eol = stdout.indexOf('\n', idx);
  return eol < 0 ? '' : stdout.slice(eol + 1);
}

function renderScriptResultInline(body, evt) {
  /* Appends to the submit_script tool-body:
   *   1. Native script output (post-preamble) — inline, visible.
   *      This is the Stata regression table, the R summary, whatever
   *      the script actually printed. The whole point of the card.
   *   2. Action buttons row: [Open in R/Stata] [Show folder].
   *
   * No sanitized-payload dropdown, no run_dir path, no stderr
   * surfacing, no error banner. Errors are explained by Claude in
   * the chat text that follows. Researchers who want the sanitized
   * payload can ask Claude to show it.
   */
  const nativeStdout = stripPreamble(evt.raw_stdout || '', evt.language).trim();
  if (nativeStdout) {
    const pre = document.createElement('pre');
    pre.className = 'tool-output';
    pre.textContent = nativeStdout;
    body.appendChild(pre);
  }

  if (evt.run_dir) {
    const actions = document.createElement('div');
    actions.className = 'tool-actions';

    const lang = evt.language;  // "R" | "Stata" | undefined
    const scriptFile = lang === 'Stata' ? 'script.do' : 'script.R';
    const openInLabel =
      lang === 'Stata' ? 'Open in Stata'
        : lang === 'R' ? 'Open in R'
        : 'Open script';
    const openMode =
      lang === 'Stata' ? 'run_stata'
        : lang === 'R' ? 'run_r'
        : null;

    const openScriptBtn = document.createElement('button');
    openScriptBtn.type = 'button';
    openScriptBtn.className = 'tool-action';
    openScriptBtn.textContent = openInLabel;
    openScriptBtn.title =
      lang === 'Stata'
        ? 'Launch Stata with the script loaded.'
        : lang === 'R'
        ? 'Launch RStudio with the script loaded.'
        : 'Open the R or Stata script in its default app.';
    openScriptBtn.addEventListener('click', () => {
      const primary = evt.run_dir + '/' + scriptFile;
      const fallback = evt.run_dir + '/' + (scriptFile === 'script.R' ? 'script.do' : 'script.R');
      openInNativeApp(primary, openScriptBtn, fallback, openMode);
    });
    actions.appendChild(openScriptBtn);

    const openFolderBtn = document.createElement('button');
    openFolderBtn.type = 'button';
    openFolderBtn.className = 'tool-action';
    openFolderBtn.textContent = 'Show folder';
    openFolderBtn.title = 'Reveal the run directory in Finder.';
    openFolderBtn.addEventListener('click', () =>
      openInNativeApp(evt.run_dir, openFolderBtn, null, null)
    );
    actions.appendChild(openFolderBtn);

    body.appendChild(actions);
  }
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
  // mcp__nora__submit_script → submit_script
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

function setWelcomeOnlyMode(enabled) {
  if (!messagesEl) return;
  messagesEl.classList.toggle('welcome-only', !!enabled);
}

// ----- policy chip + popup (next to composer) ----------------------------

const policyChip = document.getElementById('policy-chip');
const policyChipLabel = document.getElementById('policy-chip-label');
const policyPopup = document.getElementById('policy-popup');
let policyPopupBuiltFor = null;  // cached copy so we don't rebuild needlessly

function updateContextChip(inputTokens) {
  /* Updates the "Context X / Y (Z%)" chip below the composer.
   * ``inputTokens`` is the conversation-context size reported in the
   * latest ResultMessage.usage — already cumulative, since the whole
   * conversation is re-sent each turn. Unhides the chip on first
   * update and scales the ceiling up if the observed usage exceeds
   * the default 200k window (Opus 4.7 1M variant, Sonnet 4.6, etc.)
   * so the ratio stays meaningful. */
  if (!contextChip) return;
  if (typeof inputTokens !== 'number' || inputTokens < 0) return;

  if (inputTokens > contextWindow) {
    // Jumped beyond the assumed window — must be a larger-context
    // model. Round up to the next sensible tier so the ratio looks
    // stable across turns instead of creeping upward.
    contextWindow = inputTokens <= 1_000_000 ? 1_000_000 : 2_000_000;
  }

  const fmt = (n) => (n >= 10_000 ? (n / 1000).toFixed(1) + 'k' : n.toString());
  const ceilingLabel = contextWindow >= 1_000_000
    ? (contextWindow / 1_000_000) + 'M'
    : (contextWindow / 1000) + 'k';
  const pct = Math.min(100, Math.round((inputTokens / contextWindow) * 100));
  contextChip.textContent = `Context ${fmt(inputTokens)} / ${ceilingLabel} (${pct}%)`;
  contextChip.classList.remove('hidden');

  // Visual warning as context fills up — dim at low use, warm as it
  // climbs, red near the ceiling. Gives the researcher a chance to
  // wrap up a thread before the turn that exceeds the window fails.
  contextChip.classList.remove('warn', 'danger');
  if (pct >= 90) contextChip.classList.add('danger');
  else if (pct >= 70) contextChip.classList.add('warn');
}

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
  // Chip label. "Permission" reads more clearly than the internal
  // "Policy" term — it's what Claude is *permitted* to see, not a
  // legal/admin policy. Count only surfaces when something's been
  // customized; at default we keep the chip to a single word.
  const customized = policy.datasets.filter((d) => d.explicit).length;
  if (customized === 0) return 'Permission';
  return `Permission · ${customized} custom`;
}

function buildPolicyPopup(policy) {
  /* Custom popup — each dataset gets a tier-selector built from
   * buttons instead of <select>. The native dropdown ignores most
   * CSS (it uses OS chrome), so the old version went off-theme in
   * dark mode. Button rows follow the Nora palette and are
   * accessible via keyboard and screen readers. */
  const wrapper = document.createElement('div');
  const header = document.createElement('div');
  header.className = 'policy-popup-header';
  // Compact but informative: names the control, the unit it acts on,
  // and the one-way semantic (ceiling, not target). Drops the
  // "default: …" crutch — the active tier is visible in the row
  // selection itself.
  header.innerHTML =
    '<strong>Permission</strong> — ceiling on variable details ' +
    'Claude sees, per dataset. It can ask for less, never more.';
  wrapper.appendChild(header);

  policy.datasets.forEach((d) => {
    const group = document.createElement('div');
    group.className = 'policy-dataset';

    const name = document.createElement('div');
    name.className = 'policy-dataset-name';
    name.textContent = d.name;
    group.appendChild(name);

    const options = document.createElement('div');
    options.className = 'policy-options';

    DEPTH_TIERS.forEach((tier) => {
      const opt = document.createElement('button');
      opt.type = 'button';
      opt.className = 'policy-option';
      opt.dataset.value = tier.value;
      if (tier.value === d.ceiling) opt.classList.add('selected');

      const check = document.createElement('span');
      check.className = 'policy-option-check';
      check.setAttribute('aria-hidden', 'true');
      check.textContent = tier.value === d.ceiling ? '●' : '○';
      opt.appendChild(check);

      const label = document.createElement('span');
      label.className = 'policy-option-label';
      label.textContent = tier.label;
      opt.appendChild(label);

      opt.addEventListener('click', async () => {
        if (opt.classList.contains('selected')) return;
        const prev = group.querySelector('.policy-option.selected');
        // Optimistic UI: flip selection immediately; revert on error.
        options.querySelectorAll('.policy-option').forEach((el) => {
          el.classList.remove('selected');
          const c = el.querySelector('.policy-option-check');
          if (c) c.textContent = '○';
        });
        opt.classList.add('selected');
        const c = opt.querySelector('.policy-option-check');
        if (c) c.textContent = '●';

        try {
          const result = await window.pywebview.api.set_dataset_policy(
            d.name, tier.value
          );
          if (!result || !result.ok) {
            // Revert on failure.
            opt.classList.remove('selected');
            if (c) c.textContent = '○';
            if (prev) {
              prev.classList.add('selected');
              const pc = prev.querySelector('.policy-option-check');
              if (pc) pc.textContent = '●';
            }
            const err = document.createElement('div');
            err.className = 'policy-row-err';
            err.textContent = result && result.reason ? result.reason : 'failed';
            group.appendChild(err);
            setTimeout(() => err.remove(), 4000);
          } else if (result.policy) {
            updatePolicyChip(result.policy);
          }
        } catch (e) {
          opt.classList.remove('selected');
          if (c) c.textContent = '○';
          if (prev) {
            prev.classList.add('selected');
            const pc = prev.querySelector('.policy-option-check');
            if (pc) pc.textContent = '●';
          }
        }
      });
      options.appendChild(opt);
    });

    group.appendChild(options);
    wrapper.appendChild(group);
  });

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

// ----- model picker ------------------------------------------------------

const modelChip = document.getElementById('model-chip');
const modelChipLabel = document.getElementById('model-chip-label');
const modelPopup = document.getElementById('model-popup');
let availableModels = [];      // cached from list_models
let currentModelId = null;     // the backend's authoritative selection

async function loadModels() {
  /* Fetch the model catalog from the backend, render the popup,
   * and apply any locally-stored preference (from a prior session).
   */
  if (!modelChip || !window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.list_models !== 'function') return;
  try {
    const res = await window.pywebview.api.list_models();
    if (!res || !res.ok) return;
    availableModels = res.models || [];
    currentModelId = res.current;

    // If the researcher picked a model in a prior session, honor it
    // on this boot — unless it's already the current one.
    let stored = null;
    try { stored = localStorage.getItem(MODEL_STORAGE_KEY); } catch (_) {}
    if (stored && stored !== currentModelId
        && availableModels.some((m) => m.id === stored)) {
      await setModel(stored, /*silent=*/true);
    } else {
      renderModelChip();
    }
  } catch (err) {
    console.warn('list_models failed', err);
  }
}

function renderModelChip() {
  if (!modelChip || !modelChipLabel) return;
  const info = availableModels.find((m) => m.id === currentModelId);
  modelChipLabel.textContent = info ? info.label : 'Model';
  // Keep the context-chip ceiling in sync with the selected model
  // so the X / Y ratio reflects that model's actual window.
  if (info && info.context_window) {
    contextWindow = info.context_window;
  }
  renderModelPopup();
}

const PRICING_THRESHOLD = 200_000;

function renderModelPopup() {
  if (!modelPopup) return;
  modelPopup.innerHTML = '';
  const wrap = document.createElement('div');
  const header = document.createElement('div');
  header.className = 'policy-popup-header';
  header.innerHTML = '<strong>Model</strong> — Claude model Nora uses for this session.';
  wrap.appendChild(header);

  availableModels.forEach((m) => {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'model-option';
    if (m.id === currentModelId) row.classList.add('active');

    // Short hover tooltip — a punchy pricing note that appears
    // while the researcher is scanning options, before they click.
    // Few words each so the tooltip doesn't cover neighbouring
    // options. Custom ::after on the row avoids the native
    // `title`'s 500ms delay.
    if (m.context_window > PRICING_THRESHOLD) {
      row.dataset.tooltip = '~2× cost past 200k';
    } else {
      row.dataset.tooltip = 'Flat rate';
    }

    const name = document.createElement('span');
    name.className = 'model-option-name';
    name.textContent = m.label;
    row.appendChild(name);
    const ctx = document.createElement('span');
    ctx.className = 'model-option-ctx';
    ctx.textContent = formatContextWindow(m.context_window);
    row.appendChild(ctx);
    row.addEventListener('click', () => {
      modelPopup.classList.add('hidden');
      modelChip.classList.remove('open');
      setModel(m.id, /*silent=*/false);
    });
    wrap.appendChild(row);
  });

  modelPopup.appendChild(wrap);
}

function formatContextWindow(n) {
  if (!n) return '';
  if (n >= 1_000_000) return (n / 1_000_000) + 'M ctx';
  return (n / 1000) + 'k ctx';
}

async function setModel(modelId, silent) {
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.set_model !== 'function') {
    toast('Restart Nora to enable model switching.', 'error', 'model');
    return;
  }
  try {
    const res = await window.pywebview.api.set_model(modelId);
    if (!res || !res.ok) {
      if (!silent) {
        const reason = res && res.reason ? res.reason : 'unknown';
        toast('Model switch failed: ' + reason, 'error', 'model');
      }
      return;
    }
    currentModelId = modelId;
    try { localStorage.setItem(MODEL_STORAGE_KEY, modelId); } catch (_) {}
    renderModelChip();
    if (!silent && !res.unchanged) {
      toast('Model switched to ' + (res.label || modelId) + '. Takes effect on the next message.', 'success', 'model');
    }
  } catch (err) {
    console.warn('set_model failed', err);
    if (!silent) toast('Model switch failed: ' + err, 'error', 'model');
  }
}

if (modelChip && modelPopup) {
  modelChip.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOpen = !modelPopup.classList.contains('hidden');
    modelPopup.classList.toggle('hidden');
    modelChip.classList.toggle('open', !isOpen);
  });
  document.addEventListener('click', (e) => {
    if (modelPopup.classList.contains('hidden')) return;
    if (modelPopup.contains(e.target) || modelChip.contains(e.target)) return;
    modelPopup.classList.add('hidden');
    modelChip.classList.remove('open');
  });
}

// ----- session browser sidebar -------------------------------------------

const sidebarEl = document.getElementById('sidebar');
const sidebarListEl = document.getElementById('sidebar-list');
const sidebarToggleBtn = document.getElementById('sidebar-toggle');
const newSessionBtn = document.getElementById('new-session-btn');
const sidebarResizeEl = document.getElementById('sidebar-resize');
const SIDEBAR_COLLAPSED_KEY = 'nora.sidebarCollapsed';
const SIDEBAR_WIDTH_KEY = 'nora.sidebarWidth';
const SIDEBAR_MIN_WIDTH = 220;
const SIDEBAR_MAX_WIDTH = 520;

function setSidebarWidth(px, persist) {
  /* Write the width through a CSS custom property so flex-basis,
   * width, and max-width all update in one place (see .sidebar in
   * style.css). Clamped so a wild drag can't eat the chat column
   * or shrink the rail below usable size. */
  const clamped = Math.max(SIDEBAR_MIN_WIDTH, Math.min(SIDEBAR_MAX_WIDTH, px));
  if (sidebarEl) {
    sidebarEl.style.setProperty('--sidebar-width', clamped + 'px');
  }
  if (persist) {
    try { localStorage.setItem(SIDEBAR_WIDTH_KEY, String(clamped)); }
    catch (_) {}
  }
  return clamped;
}

// Restore width from a prior session before the first paint.
try {
  const stored = parseInt(localStorage.getItem(SIDEBAR_WIDTH_KEY) || '', 10);
  if (!Number.isNaN(stored)) setSidebarWidth(stored, /*persist=*/false);
} catch (_) {}

// Drag-to-resize. The handle is a 6px strip glued to the right edge.
// mousemove/up live on document so fast drags don't lose the pointer.
if (sidebarResizeEl) {
  let dragging = false;
  const onMouseMove = (e) => {
    if (!dragging) return;
    // Mouse X from the left edge of the viewport is the new width
    // (sidebar is pinned to the left). Don't persist mid-drag; wait
    // for mouseup so we only write once per gesture.
    setSidebarWidth(e.clientX, /*persist=*/false);
  };
  const onMouseUp = () => {
    if (!dragging) return;
    dragging = false;
    sidebarResizeEl.classList.remove('dragging');
    document.body.style.userSelect = '';
    document.body.style.cursor = '';
    // Persist the final width once.
    const w = parseInt(getComputedStyle(sidebarEl).width, 10);
    if (!Number.isNaN(w)) {
      try { localStorage.setItem(SIDEBAR_WIDTH_KEY, String(w)); } catch (_) {}
    }
  };
  sidebarResizeEl.addEventListener('mousedown', (e) => {
    e.preventDefault();
    dragging = true;
    sidebarResizeEl.classList.add('dragging');
    // Suppress text-selection flashing while dragging, and keep the
    // resize cursor even when the pointer strays onto other elements.
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'ew-resize';
  });
  document.addEventListener('mousemove', onMouseMove);
  document.addEventListener('mouseup', onMouseUp);
}

async function loadSessions() {
  /* Fetch the session list from the backend and render it. Called
   * on chat-view entry and after a session switch.
   *
   * If the bridge method is missing (running against an older
   * backend that hasn't been restarted since switch_session /
   * list_sessions were added), render an explanatory empty state
   * instead of a silently-blank sidebar.
   */
  if (!sidebarListEl) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.list_sessions !== 'function') {
    renderSidebarMessage('Restart Nora to see past sessions.');
    return;
  }
  try {
    const res = await window.pywebview.api.list_sessions();
    if (!res || !res.ok) {
      renderSidebarMessage('Could not load sessions.');
      return;
    }
    renderSessions(res.sessions || [], res.current || null);
  } catch (err) {
    console.warn('list_sessions failed', err);
    renderSidebarMessage('Could not load sessions.');
  }
}

function renderSidebarMessage(text) {
  if (!sidebarListEl) return;
  sidebarListEl.innerHTML = '';
  const note = document.createElement('div');
  note.className = 'sidebar-list-empty';
  note.textContent = text;
  sidebarListEl.appendChild(note);
}

function renderSessions(sessions, currentPath) {
  sidebarListEl.innerHTML = '';
  if (sessions.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'sidebar-list-empty';
    empty.textContent = 'No past sessions yet.';
    sidebarListEl.appendChild(empty);
    return;
  }
  sessions.forEach((s) => {
    // Each row: the clickable session body on the left (switches
    // cwd), a trash button on the right (deletes). The trash button
    // is a sibling rather than a nested child so its click doesn't
    // bubble to the switch handler.
    const row = document.createElement('div');
    row.className = 'session-row';
    if (s.path === currentPath) row.classList.add('active');
    row.title = s.path;

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'session-item';
    const when = document.createElement('div');
    when.className = 'session-when';
    when.textContent = formatSessionWhen(s.timestamp);
    btn.appendChild(when);

    const meta = document.createElement('div');
    meta.className = 'session-datasets';
    const dsText = s.datasets.length
      ? s.datasets.join(', ')
      : '(no data files)';
    const sizeText = typeof s.size === 'number' ? formatBytes(s.size) : '';
    meta.textContent = sizeText ? `${sizeText} · ${dsText}` : dsText;
    btn.appendChild(meta);

    btn.addEventListener('click', () =>
      switchSession(s.path, s.path === currentPath)
    );
    row.appendChild(btn);

    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'session-delete';
    del.setAttribute('aria-label', 'Delete session');
    del.title = 'Delete this session (data + history)';
    del.textContent = '×';
    del.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteSession(s, s.path === currentPath);
    });
    row.appendChild(del);

    sidebarListEl.appendChild(row);
  });
}

function formatBytes(n) {
  if (!n) return '0 B';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(0) + ' KB';
  if (n < 1024 * 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + ' MB';
  return (n / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}

async function deleteSession(s, isCurrent) {
  if (isCurrent) {
    toast('Cannot delete the active session. Switch to another first.', 'error');
    return;
  }
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.delete_session !== 'function') {
    toast('Restart Nora to enable session deletion.', 'error');
    return;
  }
  const when = formatSessionWhen(s.timestamp);
  const sizeText = typeof s.size === 'number' ? ' (' + formatBytes(s.size) + ')' : '';
  const ok = window.confirm(
    `Delete session from ${when}${sizeText}?\n\nThis removes the data copies, run logs, results.db, and chat history. Cannot be undone.`
  );
  if (!ok) return;
  try {
    const res = await window.pywebview.api.delete_session(s.path);
    if (!res || !res.ok) {
      const reason = res && res.reason ? res.reason : 'unknown';
      toast('Delete failed: ' + reason, 'error');
      return;
    }
    toast('Session deleted.', 'success');
    loadSessions();
  } catch (err) {
    console.warn('delete_session failed', err);
    toast('Delete failed: ' + err, 'error');
  }
}

function formatSessionWhen(epochSeconds) {
  /* Human-friendly timestamp for a past session.
   * Today: "3:14 PM"
   * This year: "Apr 22, 3:14 PM"
   * Older: "Apr 22, 2025"
   */
  if (!epochSeconds) return '—';
  const d = new Date(epochSeconds * 1000);
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  const sameYear = d.getFullYear() === now.getFullYear();
  const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  const date = d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  if (sameDay) return time;
  if (sameYear) return `${date}, ${time}`;
  return `${date}, ${d.getFullYear()}`;
}

async function switchSession(path, isCurrent) {
  if (isCurrent) return;  // already on it — click is a no-op
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.switch_session !== 'function') {
    // Bridge method isn't registered — almost always because the
    // backend process hasn't been restarted since this code landed.
    toast('Restart Nora to enable session switching.', 'info');
    return;
  }
  try {
    const res = await window.pywebview.api.switch_session(path);
    if (!res || !res.ok) {
      const reason = res && res.reason ? res.reason : 'unknown';
      toast('Session switch failed: ' + reason, 'error');
      return;
    }
    // showChat handles everything: hide landing, reveal chat, reset
    // transcript, refresh cwd pill + policy + sidebar + model chip,
    // hide the context chip, replay persisted history.
    showChat(res);
  } catch (err) {
    console.warn('switch_session failed', err);
    toast('Session switch failed: ' + (err && err.message ? err.message : err), 'error');
  }
}

// Arrow-key navigation inside the session list. When focus is on a
// session row, up/down moves between rows, Enter opens it, and
// Delete / Backspace deletes it (same flow as clicking the × icon).
if (sidebarListEl) {
  sidebarListEl.addEventListener('keydown', (e) => {
    const focused = document.activeElement;
    if (!focused || !sidebarListEl.contains(focused)) return;
    const buttons = Array.from(
      sidebarListEl.querySelectorAll('.session-item')
    );
    if (buttons.length === 0) return;
    // Walk up from whatever inner element has focus to the button.
    const currentBtn = focused.closest('.session-item');
    const idx = buttons.indexOf(currentBtn);

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      const next = buttons[Math.min(idx + 1, buttons.length - 1)] || buttons[0];
      if (next) next.focus();
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      const prev = buttons[Math.max(idx - 1, 0)] || buttons[0];
      if (prev) prev.focus();
      return;
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (!currentBtn) return;
      e.preventDefault();
      // Walk up to the .session-row, then click the × sibling so we
      // go through the same confirm+toast path as the mouse.
      const row = currentBtn.closest('.session-row');
      const del = row ? row.querySelector('.session-delete') : null;
      if (del) del.click();
    }
  });
}

if (sidebarToggleBtn) {
  sidebarToggleBtn.addEventListener('click', () => {
    const collapsed = sidebarEl.classList.toggle('collapsed');
    try { localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? '1' : '0'); }
    catch (_) {}
  });
  // Restore collapsed state on load.
  try {
    if (localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === '1') {
      sidebarEl.classList.add('collapsed');
    }
  } catch (_) {}
}

if (newSessionBtn) {
  newSessionBtn.addEventListener('click', () => {
    // Kick the user back to the landing screen so they can drop or
    // pick files for a brand-new session. The existing cwd stays
    // set on the backend until they finish staging, which is fine —
    // ui_ready will update on the next showChat call.
    showLanding();
  });
}

// ----- add files to current session --------------------------------------
// Lets a researcher drop more data into the active session without
// starting over. Uses the same native file-picker as landing, then
// copies into the current cwd on the backend.

const addFilesBtn = document.getElementById('add-files-btn');
if (addFilesBtn) {
  addFilesBtn.addEventListener('click', async () => {
    if (!window.pywebview || !window.pywebview.api) return;
    if (typeof window.pywebview.api.add_files !== 'function') {
      toast('Restart Nora to enable mid-session file adding.', 'info');
      return;
    }
    addFilesBtn.disabled = true;
    try {
      const res = await window.pywebview.api.add_files();
      if (!res || !res.ok) {
        const reason = res && res.reason ? res.reason : 'unknown';
        if (reason !== 'cancelled') {
          // Errors go into the main chat transcript, not a toast.
          // The researcher is looking at the conversation; a
          // message in-flow is clearer than a floating bubble.
          appendError(friendlyAddFilesError(reason));
        }
        return;
      }
      const added = res.added || [];
      const images = res.images || [];
      const skipped = res.skipped || [];

      // Stage any images the researcher picked as attachments on
      // the composer's next message. Images don't go into the
      // session cwd — they travel with the outgoing user message
      // so Claude can see them via vision.
      images.forEach((img) => {
        const url = dataUrlFromBase64(img.data, img.mime);
        stagedImages.push({ data: img.data, mime: img.mime, url });
      });
      if (images.length > 0) renderAttachments();

      // Summary toast: describe what happened in one line.
      const parts = [];
      if (added.length === 1) parts.push('Added ' + added[0]);
      else if (added.length > 1) parts.push('Added ' + added.length + ' data files');
      if (images.length === 1) parts.push('attached 1 image');
      else if (images.length > 1) parts.push('attached ' + images.length + ' images');
      if (parts.length > 0) {
        toast(parts.join(' · '), 'success');
      }
      if (skipped.length > 0) {
        toast('Skipped: ' + skipped.join(', '), 'info');
      }

      // Refresh permission chip (new data files get default policy)
      // and session title. Sidebar size also updates.
      if (res.policy) updatePolicyChip(res.policy);
      if (res.session_title && cwdEl) {
        cwdEl.textContent = res.session_title;
      }
      loadSessions();
    } catch (err) {
      console.warn('add_files failed', err);
      appendError(friendlyAddFilesError(err && err.message ? err.message : String(err)));
    } finally {
      addFilesBtn.disabled = false;
    }
  });
}

// Turn the raw error reason from the Python bridge into something
// a researcher can act on. We strip the pywebview "dialog error:"
// prefix and the regex-complaint tail, which never help the user —
// if they see "not a valid file filter" they can't fix that.
function friendlyAddFilesError(raw) {
  const s = String(raw || 'unknown error').trim();
  if (/not a valid file filter/i.test(s)) {
    return "Couldn't open the file picker. Try restarting Nora.";
  }
  if (/no active session/i.test(s)) {
    return 'Start a session first — drop files or pick a folder from the landing screen.';
  }
  if (/window not ready/i.test(s)) {
    return 'Nora is still starting up. Try again in a moment.';
  }
  // Strip the "dialog error:" prefix pywebview bubbles up.
  const cleaned = s.replace(/^dialog error:\s*/i, '');
  return "Couldn't add files: " + cleaned;
}

function dataUrlFromBase64(b64, mime) {
  // Reconstruct a data URL for <img src=...>. Browsers accept
  // either inline data URLs or blob URLs; data URL is simplest
  // when we already have the base64 string on hand.
  return `data:${mime || 'image/png'};base64,${b64}`;
}

// ----- toasts -------------------------------------------------------------
// Transient notifications that belong OUTSIDE the chat transcript:
// session switched, delete failed, model switched, "restart to
// enable X", etc. Auto-dismiss after 4 seconds, click to dismiss
// early. Chat-flow errors (turn_error, auth_failure) still go
// through appendError() so they land in the transcript alongside
// the turn they belong to.

// Single centered status line above the composer. Replaces the
// older anchored-bubble toasts that popped up in different spots
// depending on which chip fired them — researchers read those as
// "random popups". Every transient notice now lands in the same
// strip as plain dim text and auto-clears after 4s. The `anchor`
// argument is accepted for backwards-compat but ignored; everything
// flows through this one channel.
const statusLineEl = document.getElementById('status-line');
let statusClearTimer = null;

function toast(message, kind /*, anchor */) {
  if (!statusLineEl) { console.log('[status]', kind, message); return; }
  // Rebuild classes so kind-tinting doesn't accumulate across notices.
  statusLineEl.className = 'status-line visible ' + (kind || 'info');
  statusLineEl.textContent = message;
  if (statusClearTimer) {
    clearTimeout(statusClearTimer);
    statusClearTimer = null;
  }
  statusClearTimer = setTimeout(() => {
    statusLineEl.classList.remove('visible');
    // Leave text in place during the fade — the next notice
    // replaces it anyway, and empty content after a fade looks
    // abrupt.
    statusClearTimer = null;
  }, 4000);
}

// Kept so callers that imported this helper don't break; with the
// single-line design there's no bubble to dismiss.
function dismissToast() { /* no-op */ }

// ----- keyboard shortcuts ------------------------------------------------

const shortcutsOverlay = document.getElementById('shortcuts-overlay');
const shortcutsCloseBtn = document.getElementById('shortcuts-close');

function openShortcuts() {
  if (shortcutsOverlay) shortcutsOverlay.classList.remove('hidden');
}
function closeShortcuts() {
  if (shortcutsOverlay) shortcutsOverlay.classList.add('hidden');
}
if (shortcutsCloseBtn) shortcutsCloseBtn.addEventListener('click', closeShortcuts);
if (shortcutsOverlay) {
  shortcutsOverlay.addEventListener('click', (e) => {
    if (e.target === shortcutsOverlay) closeShortcuts();
  });
}

function isTypingInField(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable;
}

document.addEventListener('keydown', (e) => {
  const cmd = e.metaKey || e.ctrlKey;

  // Cmd-R: reload. Works from anywhere including the composer.
  if (cmd && e.key.toLowerCase() === 'r' && !e.shiftKey && !e.altKey) {
    e.preventDefault();
    location.reload();
    return;
  }

  // Cmd-K: focus composer.
  if (cmd && e.key.toLowerCase() === 'k' && !e.shiftKey && !e.altKey) {
    e.preventDefault();
    if (input) { input.focus(); input.select?.(); }
    return;
  }

  // Escape: close the shortcuts overlay or any open popup.
  if (e.key === 'Escape') {
    if (shortcutsOverlay && !shortcutsOverlay.classList.contains('hidden')) {
      closeShortcuts();
      return;
    }
    if (policyPopup && !policyPopup.classList.contains('hidden')) {
      policyPopup.classList.add('hidden');
      if (policyChip) policyChip.classList.remove('open');
      return;
    }
    if (modelPopup && !modelPopup.classList.contains('hidden')) {
      modelPopup.classList.add('hidden');
      if (modelChip) modelChip.classList.remove('open');
      return;
    }
  }

  // `?` opens the shortcuts overlay, but only when not typing in
  // the composer (otherwise a literal question mark never makes it
  // into the prompt).
  if (e.key === '?' && !isTypingInField(document.activeElement)) {
    e.preventDefault();
    openShortcuts();
    return;
  }
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
  // screen). showLanding() itself populates the recent-sessions list,
  // so researchers can resume a past session directly from boot.
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
