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
const authEl = document.getElementById('auth');
const authStatusEl = document.getElementById('auth-status');
const authContinueBtn = document.getElementById('auth-continue-btn');
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

// Session-level high-water mark for the context chip. Per-turn usage
// events can fluctuate (a tool-heavy turn reports a peak prompt; a
// follow-up plain turn reports a smaller one) but the underlying
// conversation chain only grows. Taking ``max(prev, latest)`` keeps
// the chip stable and informative — it tracks "biggest prompt this
// session has ever needed" instead of "whatever the last event said."
//
// Stored per-cwd (not as a single scalar) because ``showChat()`` runs
// on every ``ready`` event — including model swaps and reconnects on
// the SAME session, not just session switches. A scalar would get
// wiped to zero on those re-readies and the chip would visibly drop
// inside one conversation. Keying by cwd lets a re-ready restore the
// session's actual watermark, and a real switch start the new
// session at zero (or its own remembered value if revisited).
const sessionHighWaters = new Map();
let sessionContextHighWater = 0;

// ---- multi-session focus state -------------------------------------------
// The bridge runs every session as its own SessionRunner: turns in
// session A keep streaming after the researcher clicks B in the
// sidebar. The frontend tracks two pieces of state to render that
// honestly:
//
// - ``currentCwd``: the path of the session the user is currently
//   looking at. Set by ``showChat`` on first load and by
//   ``switchSession`` afterwards. Events that don't match this cwd
//   are background activity: their busy-state still updates the
//   sidebar dot, but they don't render into the focused
//   transcript.
// - ``busySessions``: a Set of cwd paths that have a turn currently
//   in flight. A focus-matched event flips the composer Send/Stop
//   icons; a background-matched event flips only the sidebar dot.
//   We add to the Set when a session's send is queued and remove
//   when a terminal event (turn_done / turn_error / auth_failure)
//   arrives for that session.
let currentCwd = null;
const busySessions = new Set();

// Persisted model choice — survives restarts. Applied on boot after
// (Model preference used to live in localStorage as a global default.
// It now lives per-session in ``.nora/session_state.json`` and is
// restored by the backend on session open — see ``_set_cwd`` /
// ``_restore_session_model_preference`` in ui.py.)

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

function showAuth(authPayload) {
  /* Reveal the auth screen and render per-provider rows from the
   * payload returned by ``ui_ready`` / ``auth_status``. Called on
   * first launch (no provider configured) and any time the
   * researcher clicks "back to auth" from the landing screen. */
  if (!authEl) return;
  authEl.classList.remove('hidden');
  if (landingEl) landingEl.classList.add('hidden');
  if (chatEl) chatEl.classList.add('hidden');
  if (authStatusEl) {
    authStatusEl.textContent = '';
    authStatusEl.className = 'auth-status';
  }
  renderAuthScreen(authPayload);
}

function renderAuthScreen(authPayload) {
  /* Update each provider row's status badge and Forget button based
   * on the auth_status payload. Continue button is enabled iff at
   * least one provider is configured. */
  if (!authEl) return;
  const status = (authPayload && authPayload.providers) || {};
  const rows = authEl.querySelectorAll('.auth-provider');
  rows.forEach((row) => {
    const provider = row.dataset.provider;
    const info = status[provider] || {};
    const statusEl = row.querySelector('[data-role="status"]');
    const forgetBtn = row.querySelector('[data-role="forget-btn"]');
    if (info.configured) {
      row.classList.add('configured');
      if (statusEl) {
        statusEl.classList.add('ok');
        statusEl.textContent = info.method === 'subscription'
          ? 'Signed in via Claude CLI'
          : 'API key stored';
      }
    } else {
      row.classList.remove('configured');
      if (statusEl) {
        statusEl.classList.remove('ok');
        statusEl.textContent = 'Not configured';
      }
    }
    if (forgetBtn) {
      forgetBtn.disabled = !info.has_keyring_entry;
    }
  });
  if (authContinueBtn) {
    authContinueBtn.disabled = !(authPayload && authPayload.any_authed);
  }
}

function setAuthStatus(text, kind) {
  if (!authStatusEl) return;
  authStatusEl.textContent = text || '';
  authStatusEl.className = 'auth-status' + (kind ? ' ' + kind : '');
}

async function loadAuthStatus() {
  if (!window.pywebview || !window.pywebview.api) return null;
  if (typeof window.pywebview.api.auth_status !== 'function') return null;
  try {
    return await window.pywebview.api.auth_status();
  } catch (err) {
    console.warn('auth_status failed', err);
    return null;
  }
}

if (authEl) {
  // Save / Forget per row.
  authEl.querySelectorAll('.auth-provider').forEach((row) => {
    const provider = row.dataset.provider;
    const input = row.querySelector('[data-role="key-input"]');
    const saveBtn = row.querySelector('[data-role="save-btn"]');
    const forgetBtn = row.querySelector('[data-role="forget-btn"]');

    if (saveBtn && input) {
      saveBtn.addEventListener('click', async () => {
        if (!window.pywebview || !window.pywebview.api) return;
        const key = (input.value || '').trim();
        if (!key) {
          setAuthStatus('Paste an API key first.', 'error');
          return;
        }
        saveBtn.disabled = true;
        try {
          const res = await window.pywebview.api.save_credential(provider, key);
          if (res && res.ok) {
            input.value = '';
            setAuthStatus(`${provider} key saved.`, 'ok');
            renderAuthScreen(res.auth);
          } else {
            const reason = (res && res.reason) || 'unknown error';
            setAuthStatus(`Save failed: ${reason}`, 'error');
          }
        } catch (err) {
          setAuthStatus('Save failed: ' + err, 'error');
        } finally {
          saveBtn.disabled = false;
        }
      });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') saveBtn.click();
      });
    }

    if (forgetBtn) {
      forgetBtn.addEventListener('click', async () => {
        if (!window.pywebview || !window.pywebview.api) return;
        if (typeof window.pywebview.api.delete_credential !== 'function') return;
        forgetBtn.disabled = true;
        try {
          const res = await window.pywebview.api.delete_credential(provider);
          if (res && res.ok) {
            setAuthStatus(`${provider} credential removed.`, 'ok');
            renderAuthScreen(res.auth);
          } else {
            setAuthStatus('Forget failed: ' + ((res && res.reason) || ''), 'error');
          }
        } catch (err) {
          setAuthStatus('Forget failed: ' + err, 'error');
        }
      });
    }
  });

  // Continue → land on the data picker (or chat if cwd already set).
  if (authContinueBtn) {
    authContinueBtn.addEventListener('click', async () => {
      if (!window.pywebview || !window.pywebview.api) return;
      try {
        const state = await window.pywebview.api.ui_ready();
        if (state && state.state === 'ready') {
          showChat(state);
        } else {
          showLanding();
        }
      } catch (err) {
        console.error('ui_ready failed after auth', err);
        showLanding();
      }
    });
  }
}

// "Manage providers" link on the landing card — explicit way to
// reach the auth screen when the researcher's already auto-detected
// (e.g., signed in via Claude CLI) but wants to add OpenAI too. The
// auth screen only appears automatically when no provider is
// configured at all.
async function openAuthScreen() {
  const auth = await loadAuthStatus();
  showAuth(auth || { providers: {}, any_authed: false });
}

const manageProvidersBtn = document.getElementById('manage-providers-btn');
if (manageProvidersBtn) {
  manageProvidersBtn.addEventListener('click', openAuthScreen);
}

function showLanding() {
  if (authEl) authEl.classList.add('hidden');
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
  if (authEl) authEl.classList.add('hidden');
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

  // Mark this session as the focused one. Subsequent
  // ``window.nora_event`` callbacks compare incoming
  // ``session_cwd`` against this — events for other sessions are
  // background activity and don't render into the visible
  // transcript (but still update the sidebar busy dot).
  currentCwd = payload.cwd || null;
  // Drop staged composer state so attachments don't leak across
  // sessions: an image staged in A but never sent must NOT ride
  // along with the next message in B. (Without this, the JS
  // ``stagedImages`` array stays populated through the focus
  // switch and the next form submit in B would inline A's
  // attachment into B's prompt.) Also revoke object URLs so the
  // blobs aren't pinned in memory.
  if (stagedImages.length > 0) {
    stagedImages.forEach((img) => {
      if (img && img.url) URL.revokeObjectURL(img.url);
    });
    stagedImages.length = 0;
  }
  stagedDataNotices.length = 0;
  renderAttachments();
  // Each session has its own files; bust the cache so the next
  // "@" doesn't offer rows from the previous session.
  invalidateMentionCache();
  if (typeof closeMentionPopup === 'function') closeMentionPopup();
  // Sync the composer state to whether THIS session is currently
  // busy. Switching to a session that's mid-turn shows Stop +
  // loading indicator immediately; switching to an idle session
  // shows Send.
  syncComposerToFocus();

  updatePolicyChip(payload.policy);
  loadSessions();
  loadModels();
  // Restore the high-water mark for THIS session. ``ready`` fires on
  // initial open, on every session switch, and ALSO on intra-session
  // events like model swap or provider reconnect — so we cannot
  // simply reset to zero here without wiping a chip that was already
  // tracking 500k+ within an active conversation. The map is keyed by
  // cwd; a fresh session with no prior high-water lands at zero.
  sessionContextHighWater = sessionHighWaters.get(currentCwd) || 0;
  if (contextChip) contextChip.classList.add('hidden');

  replayHistory();
  rotatePlaceholder();
  input.focus();
}

// Set while we're replaying a persisted chat log so appendAssistant
// skips the typewriter animation — past messages should appear all
// at once, not trickle in for several seconds per bubble.
let replayMode = false;
// Turns that yielded no visible reply artifacts. Keep them around
// just long enough for the researcher to notice the failure, then
// sweep them the next time a new message is sent so the transcript
// doesn't fill with dead-end bubbles.
let activeLiveTurn = null;
let replayTailTurn = null;
let staleTranscriptTurns = [];

function dropNodes(nodes) {
  (nodes || []).forEach((node) => {
    if (node && typeof node.remove === 'function') node.remove();
  });
}

function sweepStaleTranscriptTurns() {
  staleTranscriptTurns.forEach((turn) => dropNodes(turn));
  staleTranscriptTurns = [];
}

function queueDisposableTurn(nodes) {
  const kept = (nodes || []).filter(Boolean);
  if (kept.length > 0) staleTranscriptTurns.push(kept);
}

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
      if (replayTailTurn && !replayTailTurn.hasVisibleReply) {
        queueDisposableTurn(replayTailTurn.nodes);
      }
      replayTailTurn = null;
    }
    scrollToBottom();
    // Don't seed the chip from the replayed history. A chars/4
    // estimate undercounts the real prompt by a wide margin (it
    // misses the system prompt, tool schemas, tool result envelopes,
    // and reasoning traces), so the chip would show one number on
    // load and a much larger one the moment the first turn_done
    // arrived — exactly the "two largely different numbers" the
    // researcher noticed. Better to stay hidden until the provider
    // gives us the authoritative count on the next turn.
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
    case 'user_message': {
      if (replayTailTurn && !replayTailTurn.hasVisibleReply) {
        dropNodes(replayTailTurn.nodes);
      }
      // ``attachments`` may be a list of script filenames (the
      // backend persists the names that travelled with this
      // message) OR a legacy ``int`` count for image attachments.
      // Only the list form renders chips; the count form is for
      // older sessions and we just drop it.
      const att = Array.isArray(evt.attachments) ? evt.attachments : [];
      const userEl = appendUser(evt.text || '', att);
      replayTailTurn = { nodes: [userEl], hasVisibleReply: false };
      break;
    }
    case 'assistant_text':
      if (replayTailTurn) replayTailTurn.hasVisibleReply = true;
      appendAssistant(evt.text || '');
      break;
    case 'assistant_thinking':
      if (replayTailTurn) replayTailTurn.hasVisibleReply = true;
      appendThinking(evt.text || '');
      break;
    case 'tool_call': {
      const card = appendToolCall(evt);
      if (card && replayTailTurn) replayTailTurn.hasVisibleReply = true;
      break;
    }
    case 'tool_result': {
      const card = appendToolResult(evt);
      if (card && replayTailTurn) replayTailTurn.hasVisibleReply = true;
      break;
    }
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
  const accepted = files.filter(
    (f) => /\.(csv|tsv|dta|rds|parquet|jsonl|ndjson)$/i.test(f.name)
  );
  const rejected = files.length - accepted.length;
  if (accepted.length === 0) {
    setLandingError(
      'Drop .csv, .tsv, .dta, .rds, .parquet, or .jsonl files. '
      + 'Other types are ignored.'
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
    thumb.title = 'Click to view full size';
    thumb.style.cursor = 'zoom-in';
    thumb.addEventListener('click', () => showImageLightbox(img.url));
    wrap.appendChild(thumb);
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'compose-attachment-remove';
    rm.setAttribute('aria-label', 'Remove');
    rm.textContent = '×';
    rm.addEventListener('click', (e) => {
      // Stop propagation so the underlying thumbnail click
      // doesn't also fire the lightbox.
      e.stopPropagation();
      URL.revokeObjectURL(img.url);
      stagedImages.splice(idx, 1);
      renderAttachments();
    });
    wrap.appendChild(rm);
    attachmentsEl.appendChild(wrap);
  });
  // Named-chip notices for data/script files the researcher just
  // added. Data files are visual "yes that landed" receipts; script
  // files (.py / .do / .r / .rmd) ALSO travel with the next message
  // as a context block (see _pending_script_attachments in ui.py),
  // so the chip tooltip names that distinction.
  stagedDataNotices.forEach((name, idx) => {
    const chip = document.createElement('div');
    chip.className = 'compose-attachment compose-attachment-file';
    const ext = (name.split('.').pop() || '').toLowerCase();
    const isScript = ['py', 'do', 'r', 'rmd'].includes(ext);
    if (isScript) {
      chip.classList.add('compose-attachment-script');
      chip.title = name + ' — saved in this session and sent with your next message';
    } else {
      chip.title = name + ' — copied into the session';
    }
    const label = document.createElement('span');
    label.className = 'compose-attachment-filename';
    label.textContent = name;
    chip.appendChild(label);
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'compose-attachment-remove';
    rm.setAttribute('aria-label', 'Dismiss');
    rm.textContent = '×';
    rm.addEventListener('click', async () => {
      // Splice the JS-side notice immediately for a responsive UI.
      const removed = stagedDataNotices.splice(idx, 1)[0];
      renderAttachments();
      // Scripts ride the next message as inline context (the
      // backend stages them in _pending_script_attachments). The
      // chip × must also call the bridge to unstage there —
      // without this, the file silently rides along after the
      // researcher dismissed it. The on-disk copy is untouched.
      if (
        isScript
        && window.pywebview
        && window.pywebview.api
        && typeof window.pywebview.api.unstage_attachment === 'function'
      ) {
        try {
          await window.pywebview.api.unstage_attachment(removed);
        } catch (err) {
          console.warn('unstage_attachment failed', err);
        }
      }
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
// Same set the + button's picker accepts — data files, R/Stata/
// Python scripts, Stata graphs, logs, and R Markdown. All copied
// into the session cwd so the model can reference them.
const COMPOSER_DATA_EXTS = new Set([
  'csv', 'tsv', 'dta', 'rds', 'parquet', 'jsonl', 'ndjson',
  'do', 'r', 'py', 'ipynb',
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
    if (addStagedDataNotices(res.added || [])) {
      renderAttachments();
    }
    if (res.skipped && res.skipped.length > 0) {
      appendError('Skipped: ' + res.skipped.join(', '));
    }
    if (res.skipped_existing && res.skipped_existing.length > 0) {
      // Already-in-session collisions are surfaced as their own line so
      // it's clear nothing was overwritten. Researcher can rename
      // upstream and try again.
      appendError(
        'Already in this session, not overwritten: '
        + res.skipped_existing.join(', ')
      );
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

function addStagedDataNotices(names) {
  /* Mirror backend-staged data/script files into the composer's
   * receipt chips. Deduplicate by basename so the native Add Files
   * button, drag/drop, and Files-popup attach all keep one visual
   * chip per staged file.
   */
  let changed = false;
  (names || []).forEach((name) => {
    if (!name || stagedDataNotices.includes(name)) return;
    stagedDataNotices.push(name);
    changed = true;
  });
  return changed;
}

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
        ' — only images and data/script files (.csv, .tsv, .dta, .rds, .parquet, .jsonl, .do, .r, .py, .ipynb, .log, .smcl, .gph, .rmd) can be dropped here.'
      );
    }
    for (const file of usable) {
      if (ALLOWED_IMAGE_MIMES.has(file.type)) {
        // Stage for one-turn vision AND persist to the session cwd
        // so the model can @-mention or read_attached_file the
        // image on later turns. Earlier behavior was vision-only,
        // which made dropped images one-shot while native "+ Add
        // Files" persisted them — confusing inconsistency.
        await stageImageFile(file);
        await stageDataFile(file);
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
        // Stage for vision AND persist — same dual-tracking as the
        // drop handler so paste and drop produce identical state.
        await stageImageFile(f);
        await stageDataFile(f);
      } else {
        await stageDataFile(f);
      }
    }
  });
}

// ---- @-mention dropdown for session files --------------------------------
//
// When the researcher types "@" the composer offers a filtered list of
// every file already in this session: scripts, datasets, plots, logs.
// Selecting a row stages the file via attach_session_file (the same
// bridge endpoint the Files panel uses) and inserts "@<filename>" at
// the caret. The transcript chip + composer chip then track the file
// the same way a drag/drop attachment would.
let mentionFiles = null;
let mentionFilesFresh = false;
const mentionPopup = document.createElement('div');
mentionPopup.id = 'mention-popup';
mentionPopup.className = 'mention-popup hidden';
document.body.appendChild(mentionPopup);
let mentionState = null;

function invalidateMentionCache() {
  mentionFilesFresh = false;
}

async function ensureMentionFiles() {
  if (mentionFilesFresh && Array.isArray(mentionFiles)) return mentionFiles;
  if (!window.pywebview || !window.pywebview.api) return [];
  if (typeof window.pywebview.api.list_mentionable_files !== 'function') return [];
  try {
    const res = await window.pywebview.api.list_mentionable_files();
    if (res && res.ok && Array.isArray(res.files)) {
      mentionFiles = res.files;
      mentionFilesFresh = true;
      return mentionFiles;
    }
  } catch (err) {
    console.warn('list_mentionable_files failed', err);
  }
  mentionFiles = [];
  mentionFilesFresh = true;
  return mentionFiles;
}

function detectMentionTrigger() {
  if (!input) return null;
  const value = input.value;
  const caret = input.selectionStart;
  if (caret == null || caret !== input.selectionEnd) return null;
  let i = caret - 1;
  let scanned = 0;
  while (i >= 0) {
    const c = value[i];
    if (c === '@') {
      if (i === 0 || /\s/.test(value[i - 1])) {
        return {
          startIdx: i,
          endIdx: caret,
          query: value.slice(i + 1, caret).toLowerCase(),
        };
      }
      return null;
    }
    if (/\s/.test(c)) return null;
    scanned += 1;
    if (scanned > 64) return null;
    i -= 1;
  }
  return null;
}

function filterMentionFiles(files, query) {
  if (!query) return files.slice(0, 20);
  const scored = [];
  for (const f of files) {
    const lname = (f.name || '').toLowerCase();
    const idx = lname.indexOf(query);
    if (idx === -1) continue;
    let score = 10;
    if (idx === 0) score = 100;
    else {
      const prev = lname[idx - 1];
      if (prev === '.' || prev === '_' || prev === '-') score = 50;
    }
    score -= idx;
    scored.push({ score, f });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 20).map((s) => s.f);
}

function mentionKindIcon(kind) {
  switch (kind) {
    case 'script': return 'S';
    case 'data':   return 'D';
    case 'graph':  return 'G';
    case 'log':    return 'L';
    default:       return '·';
  }
}

function mentionFormatBytes(n) {
  if (n == null) return '';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return Math.round(n / 1024) + ' KB';
  return (n / (1024 * 1024)).toFixed(1) + ' MB';
}

function escapeMentionHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
    c === '&' ? '&amp;' :
    c === '<' ? '&lt;' :
    c === '>' ? '&gt;' :
    c === '"' ? '&quot;' : '&#39;'
  ));
}

function renderMentionPopup() {
  if (!mentionState) {
    mentionPopup.classList.add('hidden');
    return;
  }
  const { items, selected } = mentionState;
  mentionPopup.innerHTML = '';
  if (!items || items.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'mention-empty';
    empty.textContent = 'No matching files in this session.';
    mentionPopup.appendChild(empty);
  } else {
    const list = document.createElement('div');
    list.className = 'mention-list';
    items.forEach((f, idx) => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'mention-row' + (idx === selected ? ' selected' : '');
      row.dataset.idx = String(idx);
      row.innerHTML = (
        '<span class="mention-icon mention-icon-' + escapeMentionHtml(f.kind || '') + '" '
          + 'aria-hidden="true">' + escapeMentionHtml(mentionKindIcon(f.kind)) + '</span>'
        + '<span class="mention-name">' + escapeMentionHtml(f.name) + '</span>'
        + '<span class="mention-meta">' + escapeMentionHtml(f.kind || '')
          + (f.size != null ? ' · ' + escapeMentionHtml(mentionFormatBytes(f.size)) : '')
          + '</span>'
      );
      row.addEventListener('mousedown', (e) => {
        e.preventDefault();
        selectMention(idx);
      });
      row.addEventListener('mouseenter', () => {
        if (!mentionState) return;
        mentionState.selected = idx;
        list.querySelectorAll('.mention-row').forEach((el, i) => {
          el.classList.toggle('selected', i === idx);
        });
      });
      list.appendChild(row);
    });
    mentionPopup.appendChild(list);
    const hint = document.createElement('div');
    hint.className = 'mention-hint';
    hint.textContent = '↑↓ navigate · ↵ insert · esc dismiss';
    mentionPopup.appendChild(hint);
  }
  mentionPopup.classList.remove('hidden');
  positionMentionPopup();
}

function positionMentionPopup() {
  if (!input) return;
  const rect = input.getBoundingClientRect();
  const width = Math.min(Math.max(rect.width, 320), 480);
  mentionPopup.style.width = width + 'px';
  mentionPopup.style.left = rect.left + 'px';
  requestAnimationFrame(() => {
    const popupH = mentionPopup.offsetHeight;
    const top = Math.max(8, rect.top - popupH - 6);
    mentionPopup.style.top = top + 'px';
  });
}

function closeMentionPopup() {
  mentionState = null;
  mentionPopup.classList.add('hidden');
}

async function refreshMentionState() {
  const trigger = detectMentionTrigger();
  if (!trigger) {
    closeMentionPopup();
    return;
  }
  const files = await ensureMentionFiles();
  if (files.length === 0) {
    closeMentionPopup();
    return;
  }
  const items = filterMentionFiles(files, trigger.query);
  mentionState = {
    startIdx: trigger.startIdx,
    endIdx: trigger.endIdx,
    query: trigger.query,
    items,
    selected: 0,
  };
  renderMentionPopup();
}

async function selectMention(idx) {
  if (!mentionState) return;
  const item = mentionState.items[idx];
  if (!item) return;
  const before = input.value.slice(0, mentionState.startIdx);
  const after = input.value.slice(mentionState.endIdx);
  const token = '@' + item.name;
  input.value = before + token + after;
  const caret = before.length + token.length;
  input.selectionStart = input.selectionEnd = caret;
  autosize();
  closeMentionPopup();
  await stageMentionedFile(item.name);
  input.focus();
}

async function stageMentionedFile(name) {
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.attach_session_file !== 'function') return;
  try {
    const res = await window.pywebview.api.attach_session_file(name);
    if (!res || !res.ok) {
      const reason = (res && res.reason) || 'unknown';
      toast('Could not attach: ' + reason, 'error');
      return;
    }
    if (!res.already_attached) {
      if (addStagedDataNotices([res.name || name])) renderAttachments();
    }
  } catch (err) {
    console.warn('attach_session_file failed', err);
  }
}

if (input) {
  input.addEventListener('input', () => { refreshMentionState(); });
  input.addEventListener('click', () => { refreshMentionState(); });
  input.addEventListener('keyup', (e) => {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight'
        || e.key === 'Home' || e.key === 'End') {
      refreshMentionState();
    }
  });
  input.addEventListener('keydown', (e) => {
    if (!mentionState || mentionPopup.classList.contains('hidden')) return;
    const items = mentionState.items || [];
    const len = Math.max(items.length, 1);
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      mentionState.selected = (mentionState.selected + 1) % len;
      renderMentionPopup();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      mentionState.selected = (mentionState.selected - 1 + len) % len;
      renderMentionPopup();
    } else if (e.key === 'Enter' || e.key === 'Tab') {
      if (items.length > 0) {
        e.preventDefault();
        e.stopImmediatePropagation();
        selectMention(mentionState.selected);
      } else {
        closeMentionPopup();
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      e.stopImmediatePropagation();
      closeMentionPopup();
    }
  });
  input.addEventListener('blur', () => {
    setTimeout(closeMentionPopup, 120);
  });
}
window.addEventListener('resize', () => {
  if (mentionState) positionMentionPopup();
});

// ---- send-while-busy queue -----------------------------------------------
//
// When a turn is already in flight, the user can still type and Send.
// We render the user bubble immediately with a "queued" pill, snapshot
// the staged attachments, and fire the actual ``send_message`` only
// when the current turn's terminal event arrives. The runner's
// per-session ``_send_lock`` already serialises sends on the Python
// side; this queue is purely about (a) capturing the right
// attachments / images at submit time (so they go with the right
// message rather than getting folded into whichever turn is running)
// and (b) giving the researcher visible feedback that their follow-up
// landed.
//
// Stop drains the queue: cancelled queued bubbles get marked
// ``.not-sent`` and stay visible (with reduced opacity) so the
// researcher can see what they asked but didn't ship.
const pendingByCwd = new Map();

function pendingFor(cwd) {
  let q = pendingByCwd.get(cwd);
  if (!q) { q = []; pendingByCwd.set(cwd, q); }
  return q;
}

async function fireQueuedMessage(cwd, item) {
  item.userEl.classList.remove('queued');
  activeLiveTurn = { nodes: [item.userEl], hasVisibleReply: false };
  try {
    if (item.images.length > 0 && typeof window.pywebview.api.send_message_with_images === 'function') {
      const payload = item.images.map((img) => ({ data: img.data, mime: img.mime }));
      await window.pywebview.api.send_message_with_images(item.text, payload);
    } else if (item.images.length > 0) {
      const errEl = appendError('Restart Nora to send images.');
      if (activeLiveTurn) {
        activeLiveTurn.nodes.push(errEl);
        queueDisposableTurn(activeLiveTurn.nodes);
      }
      activeLiveTurn = null;
      setSending(false, cwd);
    } else {
      await window.pywebview.api.send_message(item.text);
    }
  } catch (err) {
    const errEl = appendError('send failed: ' + err);
    if (activeLiveTurn) {
      activeLiveTurn.nodes.push(errEl);
      queueDisposableTurn(activeLiveTurn.nodes);
    }
    activeLiveTurn = null;
    setSending(false, cwd);
  }
}

function flushPendingFor(cwd) {
  const q = pendingByCwd.get(cwd);
  if (!q || q.length === 0) return false;
  const next = q.shift();
  setSending(true, cwd);
  Promise.resolve().then(() => fireQueuedMessage(cwd, next));
  return true;
}

function drainPendingFor(cwd) {
  const q = pendingByCwd.get(cwd);
  if (!q || q.length === 0) return 0;
  const drained = q.length;
  for (const item of q) {
    item.userEl.classList.remove('queued');
    item.userEl.classList.add('not-sent');
    item.userEl.title = 'Stopped before this message could send.';
  }
  q.length = 0;
  return drained;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  const images = stagedImages.slice();  // snapshot
  // Send is allowed when ANY of {text, image, attached data file} is
  // present. The earlier guard ignored ``stagedDataNotices`` so a
  // researcher who attached a .do file and pressed Send without
  // typing got silent nothing — looked like file calling was
  // broken. With a script chip in the composer the model gets the
  // attachment as context and can proceed (the chip name itself is
  // implicit "run / inspect this").
  if (!text && images.length === 0 && stagedDataNotices.length === 0) return;
  if (!window.pywebview || !window.pywebview.api) {
    appendSystem('backend not ready yet; try again');
    return;
  }
  // Snapshot which staged-data notices are travelling with THIS
  // message (only script files actually ride along as inline context;
  // pure data files are session-resident and discoverable via
  // get_schema). Rendered as transcript chips below the user bubble
  // so the upload stays visible after send instead of disappearing
  // with the composer chip.
  const SCRIPT_EXTS_RE = /\.(py|do|r|rmd)$/i;
  const messageAttachments = stagedDataNotices.filter(
    (n) => SCRIPT_EXTS_RE.test(n)
  );
  // Snapshot image thumbnails for the user bubble. Use base64
  // data URLs (not blob:), because the form submit immediately
  // revokes the blob URLs — keeping the blob would leave the
  // bubble's thumbnails dead. data: URLs are self-contained, so
  // they survive the blob revoke at the bottom of this handler.
  const messageImages = images.map((img) => ({
    url: dataUrlFromBase64(img.data, img.mime),
    mime: img.mime,
  }));
  sweepStaleTranscriptTurns();
  const userEl = appendUser(
    text || '(image only)', messageAttachments, messageImages
  );
  input.value = '';
  closeMentionPopup();
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

  // If a turn is already running on this session, queue the new
  // message instead of firing it. The terminal-event handler
  // (turn_done / turn_error / auth_failure) drains the queue, so
  // by the time the running turn finishes the next one fires
  // automatically. The user bubble is already in the transcript;
  // we tag it ``.queued`` so the researcher can see what's
  // pending.
  if (turnInFlight) {
    userEl.classList.add('queued');
    userEl.title = 'Queued — will send when the current turn finishes.';
    pendingFor(currentCwd).push({
      text,
      images: images.map((img) => ({ data: img.data, mime: img.mime })),
      attachments: messageAttachments,
      userEl,
    });
    return;
  }

  activeLiveTurn = { nodes: [userEl], hasVisibleReply: false };
  setSending(true);
  try {
    // If images are attached, use the richer send method. The
    // simpler string send stays as the fast path for text-only.
    if (images.length > 0 && typeof window.pywebview.api.send_message_with_images === 'function') {
      const payload = images.map((img) => ({ data: img.data, mime: img.mime }));
      await window.pywebview.api.send_message_with_images(text, payload);
    } else if (images.length > 0) {
      const errEl = appendError('Restart Nora to send images.');
      if (activeLiveTurn && !activeLiveTurn.hasVisibleReply) {
        activeLiveTurn.nodes.push(errEl);
        queueDisposableTurn(activeLiveTurn.nodes);
      }
      activeLiveTurn = null;
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
    const errEl = appendError('send failed: ' + err);
    if (activeLiveTurn && !activeLiveTurn.hasVisibleReply) {
      activeLiveTurn.nodes.push(errEl);
      queueDisposableTurn(activeLiveTurn.nodes);
    }
    activeLiveTurn = null;
    setSending(false);
  }
});

// Shift-Enter inserts a newline; plain Enter sends. Sending while a
// turn is already in flight is allowed: the submit handler queues
// the new message and the terminal-event handler drains the queue.
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
  'Postcard from the ivory tower',
  'Reviewer 2 is asleep, talk to me',
  'Endogeneity hotline',
  'Heteroskedasticity-robust greetings',
  'I read code so you don\'t have to',
  'Bring me your messy joins',
  'Give me your tired, your missing, your truncated',
  'Surrender the .dta',
  'Throw your reshapes at me',
  'Panel data, biscuits, tea',
];

function rotatePlaceholder() {
  if (!input) return;
  const next = PLACEHOLDERS[Math.floor(Math.random() * PLACEHOLDERS.length)];
  input.setAttribute('placeholder', next);
}
rotatePlaceholder();

function setSending(sending, cwd) {
  // ``cwd`` defaults to the focused session — this is what
  // happens when send_message is called from the form submit.
  // Background events (terminal events for non-focused sessions)
  // pass an explicit cwd so the busy state lands on the right row.
  const target = cwd || currentCwd;
  if (target) {
    setSessionBusy(target, sending);
  }
  // Composer state mirrors the FOCUSED session only.
  if (target && target !== currentCwd) return;
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

function setSessionBusy(cwd, busy) {
  /* Track which sessions have a turn in flight so the sidebar can
   * show a small "still working" dot on background sessions. The
   * dot is purely informational — clicking the row still works as
   * a normal focus switch and the in-flight turn keeps streaming
   * regardless. */
  if (!cwd) return;
  if (busy) busySessions.add(cwd);
  else busySessions.delete(cwd);
  // Update the sidebar row in place. Falls through if the row
  // isn't currently rendered (e.g., scrolled off in a long list);
  // the next loadSessions() will re-render with the right state.
  const rows = document.querySelectorAll('.session-item');
  rows.forEach((row) => {
    if (row.dataset && row.dataset.path === cwd) {
      row.classList.toggle('busy', busy);
    }
  });
}

function syncComposerToFocus() {
  /* Called on every focus switch. Sets the composer state to match
   * whether the now-focused session has a turn in flight. Without
   * this, switching from a busy session to an idle one would leave
   * Stop on screen, and switching back to a busy session would
   * show Send (the wrong control). */
  const focusedBusy = currentCwd && busySessions.has(currentCwd);
  setSending(!!focusedBusy, currentCwd);
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

// Stop button — asks the bridge to cancel the in-flight turn AND
// always returns the UI to a clean state. The bridge cancels the
// asyncio task and tears down the SDK client so no half-finished
// request leaks into the next turn. We used to wait for the
// turn_error event to clear setSending, but the provider stream can
// in rare cases close without yielding any terminal event (network
// blip, SDK internal hiccup); the JS then stayed stuck on "sending"
// forever and the bridge said "no turn in flight." Hard-resetting
// here means Stop is always a reliable recovery button: researchers
// can always get the composer back.
if (stopBtn) {
  stopBtn.addEventListener('click', async () => {
    if (!window.pywebview || !window.pywebview.api) return;
    stopBtn.disabled = true;
    // Stop = "stop everything for this session": cancel the
    // running turn AND drain any queued follow-ups. Cancelled
    // queued messages get marked ``.not-sent`` so the researcher
    // can see what they typed but didn't ship; silently removing
    // their text would be hostile UX.
    drainPendingFor(currentCwd);
    // Visible acknowledgement: the cancellation cascades through
    // the SDK + the subprocess kill, which can take a beat. Without
    // this toast, a researcher who pressed Stop and immediately
    // resent the same prompt would see the new turn queue behind
    // the cancellation cleanup and assume "Stop did nothing".
    toast('Stopping…', 'info');
    try {
      await window.pywebview.api.interrupt_turn();
    } catch (_) {
      // swallow: the bridge may have nothing to cancel; we still
      // want to clear the UI state below.
    } finally {
      // Always restore the composer regardless of what the bridge
      // said. If a terminal event arrives after this, it's a no-op
      // (setSending(false) is idempotent); if none arrives, the
      // researcher isn't trapped.
      setSending(false);
      stopBtn.disabled = false;
    }
  });
}

// ----- Python → JS event handler -----------------------------------------

window.nora_event = function (evt) {
  // evt is a plain object; {type} + type-specific fields.
  // Every event from a runner carries ``session_cwd``: the cwd of
  // the runner that emitted it. ``ready`` and ``policy_updated``
  // are bridge-level events without a session_cwd — those always
  // apply to the focused session and pass through.
  const evtCwd = evt.session_cwd;
  const isFocused = !evtCwd || (currentCwd && evtCwd === currentCwd);

  switch (evt.type) {
    case 'ready':
      showChat(evt);
      break;
    case 'assistant_text':
      if (!isFocused) return;
      if (activeLiveTurn) activeLiveTurn.hasVisibleReply = true;
      appendAssistant(evt.text);
      break;
    case 'assistant_thinking':
      if (!isFocused) return;
      if (activeLiveTurn) activeLiveTurn.hasVisibleReply = true;
      appendThinking(evt.text);
      break;
    case 'tool_call': {
      if (!isFocused) return;
      const card = appendToolCall(evt);
      if (card && activeLiveTurn) activeLiveTurn.hasVisibleReply = true;
      break;
    }
    case 'tool_result': {
      if (!isFocused) return;
      const card = appendToolResult(evt);
      if (card && activeLiveTurn) activeLiveTurn.hasVisibleReply = true;
      // A submit_script run can produce new plots that should
      // accumulate in the topbar Files panel. Refresh after each
      // tool_result so the right-corner panel stays the
      // session-wide gallery — researcher scrolls back through
      // every plot the analysis ever produced without leaving the
      // chat.
      if (evt.plots && evt.plots.length > 0) {
        refreshFilesChip();
      }
      break;
    }
    case 'turn_done':
      // Terminal event: clear busy state for THIS session
      // (whether focused or background) and, if focused, refresh
      // the composer + context chip.
      //
      // The chip tracks "context occupied AFTER this turn" — i.e.,
      // the prompt this turn loaded PLUS the response that just
      // landed. Including ``output_tokens`` means a long reply
      // shows up on the chip the instant it arrives, instead of
      // only on the next turn (when the SDK folds it back into
      // input/cache_read). The natural-language reading of "how
      // full is my chat" wants the post-turn snapshot, not the
      // pre-response one.
      if (isFocused) {
        const occupied =
          (evt.input_tokens || 0) +
          (evt.cache_read_input_tokens || 0) +
          (evt.cache_creation_input_tokens || 0) +
          (evt.output_tokens || 0);
        updateContextChip(occupied);
        if (activeLiveTurn && !activeLiveTurn.hasVisibleReply) {
          queueDisposableTurn(activeLiveTurn.nodes);
        }
        activeLiveTurn = null;
      }
      // ``flushPendingFor`` returns true iff a queued message just
      // fired. In that case ``setSending(true, evtCwd)`` was
      // re-asserted inside, so we leave the composer in the busy
      // state and DON'T flip back to Send.
      if (!flushPendingFor(evtCwd)) {
        setSending(false, evtCwd);
      }
      break;
    case 'auth_failure':
      // Auth failures matter cross-session: even a background
      // turn that hits an auth error should drop its busy dot.
      // Render the error bubble only into the focused transcript
      // (the message is in the persisted log; switching to that
      // session will replay it). Drain any queued follow-ups for
      // this session: if auth is broken, queueing them up to fail
      // one after another is just noise.
      if (isFocused) {
        const errEl = appendError('Auth failure: ' + (evt.reason || 'unknown'));
        if (activeLiveTurn && !activeLiveTurn.hasVisibleReply) {
          activeLiveTurn.nodes.push(errEl);
          queueDisposableTurn(activeLiveTurn.nodes);
        }
        activeLiveTurn = null;
      }
      drainPendingFor(evtCwd);
      setSending(false, evtCwd);
      break;
    case 'turn_error':
      if (isFocused) {
        const errEl = appendError(evt.message || 'unknown error');
        if (activeLiveTurn && !activeLiveTurn.hasVisibleReply) {
          activeLiveTurn.nodes.push(errEl);
          queueDisposableTurn(activeLiveTurn.nodes);
        }
        activeLiveTurn = null;
      }
      // For ordinary turn errors (e.g., model returned a tool-use
      // error), drain the queue: the researcher's follow-ups were
      // probably reasoning-conditioned on the previous turn
      // succeeding, so firing them blindly is worse than asking
      // them to retry.
      drainPendingFor(evtCwd);
      setSending(false, evtCwd);
      break;
    case 'policy_updated':
      updatePolicyChip(evt.policy);
      break;
    default:
      console.warn('unknown event type', evt);
  }
};

// ----- message rendering --------------------------------------------------

function appendUser(text, attachments, images) {
  /* ``attachments`` is an optional array of filenames that traveled
   * with this message (e.g. dragged-in .py / .do scripts).
   * ``images`` is an optional array of ``{url, mime}`` data-URL
   * thumbnails. Both render in the transcript so the upload event
   * is visible permanently — not just as ephemeral composer chips
   * that clear on send. */
  return append('user', text, /*markdown=*/ false, attachments || [], images || []);
}

function appendAssistant(text) {
  // Render markdown immediately for both replay and live paths. The
  // earlier typewriter animation cushioned wall-of-text shock but
  // showed RAW markdown during typing — tables as ``|---|---|``,
  // code blocks as escaped text, bold as literal asterisks — and
  // delayed the rendered version by up to ~37 s on long replies.
  // For Nora's audience (dense regression tables, stata/python
  // code, multi-section answers) the cost compounded: the researcher
  // couldn't read the structure that mattered most until the swap.
  // Going without typing animation; revisit if it turns out the
  // chat metaphor needed it.
  //
  // The runTypewriter / finalizeActiveTypewriter scaffolding still
  // lives below in case we re-enable a (skippable) version of this.
  return append('assistant', text || '', /*markdown=*/ true);
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
  // Tiered constant pace. Base rate is ~22 ms per char (~45 chars/sec),
  // a thoughtful typing rhythm rather than a stream. To keep long
  // messages from trapping the reader, the rate steps up at two length
  // thresholds: 2× past 500 chars, 3× past 2000. It never goes faster
  // than 3× — past 2000 chars the animation just takes longer, the way
  // a constant-pace typewriter naturally would.
  //
  // Earlier code clamped total animation time to ~4 s, which made
  // ``charsPerMs = len/4000`` grow linearly with length (a 5000-char
  // turn typed at ~1250 cps, indistinguishable from an instant dump).
  // The tiered approach gives the same "doesn't make me wait forever"
  // behavior on truly long turns without erasing the typing rhythm
  // entirely the moment a message crosses some arbitrary length.
  //
  // Any follow-up UI event (next assistant block, tool call, thinking,
  // user input) calls ``finalizeActiveTypewriter()`` and collapses the
  // animation immediately — see its call sites.
  const speedFactor = len < 500 ? 1 : len < 2000 ? 2 : 3;
  const MS_PER_CHAR = 22 / speedFactor;
  let typed = 0;
  // Fractional accumulator. Per-frame ``Math.ceil`` (the earlier
  // approach) forced at least 1 char every animation frame, which at
  // 60 fps clamps the floor to ~60 cps — the documented 45 cps target
  // was unreachable. Accumulating fractional progress lets us advance
  // 0 chars on some frames so the visible rate matches MS_PER_CHAR.
  let accumulator = 0;
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
    accumulator += dt / MS_PER_CHAR;
    const advance = Math.floor(accumulator);
    if (advance > 0) {
      typed = Math.min(len, typed + advance);
      accumulator -= advance;
      bodyEl.textContent = fullText.slice(0, typed);
      scrollToBottom();
    }
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
  return card;
}

function appendSystem(text) {
  return append('system', text);
}

function appendError(text) {
  return append('error', text);
}

function append(kind, text, markdown, attachments, images) {
  // User / system / error messages appear instantly. Make sure any
  // typewriter from the previous turn lands first, so a fresh user
  // bubble doesn't appear above a still-animating assistant bubble.
  setWelcomeOnlyMode(false);
  finalizeActiveTypewriter();
  const wrapper = document.createElement('div');
  wrapper.className = 'message ' + kind;
  // Image thumbnails render ABOVE the bubble — same vertical order
  // they appeared in the composer, easier to scan. Clicking opens
  // the full-resolution image in a new browser tab so the
  // researcher can inspect details (axis labels, fine print, etc.)
  // that don't survive the chat-width thumbnail size.
  if (images && images.length > 0) {
    const row = document.createElement('div');
    row.className = 'message-images';
    images.forEach((img, idx) => {
      const thumb = document.createElement('img');
      thumb.src = img.url || '';
      thumb.alt = `Attached image ${idx + 1}`;
      thumb.className = 'message-image-thumb';
      thumb.title = 'Click to view full size';
      thumb.addEventListener('click', () => {
        if (img.url) showImageLightbox(img.url);
      });
      row.appendChild(thumb);
    });
    wrapper.appendChild(row);
  }
  const body = document.createElement('div');
  body.className = 'message-body';
  if (markdown && window.NoraMarkdown) {
    body.innerHTML = window.NoraMarkdown.render(text);
  } else {
    body.textContent = text;
  }
  wrapper.appendChild(body);
  // Attachment chips render as a small row beneath the user bubble
  // (only shown when the caller passes a non-empty list). This is
  // the "I uploaded a script and Nora can see it" affordance the
  // composer chip can't be (composer chips clear on send).
  if (attachments && attachments.length > 0) {
    const row = document.createElement('div');
    row.className = 'message-attachments';
    attachments.forEach((name) => {
      const chip = document.createElement('span');
      chip.className = 'message-attachment-chip';
      chip.textContent = '📎 ' + name;
      chip.title = name + ' — sent with this message';
      row.appendChild(chip);
    });
    wrapper.appendChild(row);
  }
  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function appendToolCall(evt) {
  // Only ``submit_script`` and ``submit_script_file`` render cards.
  // ``get_schema``, ``request_data``, ``expand_result``,
  // ``list_results`` etc. happen silently — they're plumbing, not
  // results the researcher reads. Claude summarizes whatever
  // matters from them in the chat text that follows.
  finalizeActiveTypewriter();
  setWelcomeOnlyMode(false);
  const shortName = shortenToolName(evt.name);
  const isSubmitScript = shortName === 'submit_script';
  const isSubmitScriptFile = shortName === 'submit_script_file';
  if (!isSubmitScript && !isSubmitScriptFile) return;

  const card = document.createElement('div');
  card.dataset.callId = evt.call_id;
  card.className = 'tool-card';

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

  const input = evt.input || {};
  if (isSubmitScript) {
    // Render the language + code + label prominently, not as JSON
    // stringification. Researcher sees the actual script.
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
  } else if (isSubmitScriptFile) {
    // The script bytes don't ride in the tool input — the file is
    // already on disk in cwd. Render filename + label + language so
    // the researcher recognises which attachment is being run.
    const langText = (input.language || '').toString().toUpperCase();
    if (input.label) {
      const label = document.createElement('div');
      label.className = 'tool-label';
      label.textContent = input.label;
      body.appendChild(label);
    }
    const pre = document.createElement('pre');
    pre.className = 'tool-code lang-' + (input.language || 'text').toLowerCase();
    const badge = document.createElement('span');
    badge.className = 'tool-lang-badge';
    badge.textContent = langText || 'script';
    pre.appendChild(badge);
    const fileEl = document.createElement('code');
    fileEl.textContent = 'running ' + (input.name || '(unnamed file)');
    pre.appendChild(fileEl);
    body.appendChild(pre);
    if (input.source_dataset) {
      const src = document.createElement('div');
      src.className = 'tool-source';
      src.textContent = 'source: ' + input.source_dataset;
      body.appendChild(src);
    }
  }

  header.addEventListener('click', () => card.classList.toggle('collapsed'));

  card.appendChild(header);
  card.appendChild(body);
  messagesEl.appendChild(card);
  scrollToBottom();
  return card;
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
  return existingCard;
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
   *   1. Nora-rendered canonical result tables, when the tool
   *      result envelope carries them (one per ok-status entry's
   *      ``markdown`` field). Product output, not model prose —
   *      the same payload renders identically across recalls.
   *   2. Native script output (post-preamble) — inline, visible.
   *      The Stata regression table, the R summary, whatever the
   *      script actually printed.
   *   3. Action buttons row: [Open in R/Stata] [Show folder].
   *
   * The model still interprets the result in chat; table SHAPE
   * (column choice, p-value column, precision) is enforced here.
   */
  renderCanonicalResultTables(body, evt);

  const nativeStdout = stripPreamble(evt.raw_stdout || '', evt.language).trim();
  if (nativeStdout) {
    // Collapsed by default. The script + the canonical regression
    // tables above already tell the researcher what they need; the
    // raw R/Stata/Python log is for "let me audit" moments. Same
    // disclosure pattern as the multi-result panel.
    const details = document.createElement('details');
    details.className = 'tool-output-collapsed';
    const summary = document.createElement('summary');
    summary.className = 'tool-output-summary';
    const lang = evt.language || 'script';
    summary.textContent = `${lang} output (click to expand)`;
    details.appendChild(summary);
    const pre = document.createElement('pre');
    pre.className = 'tool-output';
    pre.textContent = nativeStdout;
    details.appendChild(pre);
    body.appendChild(details);
  }

  // Plot-helper diagnostic — surfaced when a helper was clearly
  // called (the run-dir has a ``_nora_plots/`` subdir + stderr
  // mentions ``nora.plot_*``) but no plot files actually landed.
  // Without this note, the researcher sees an empty thumbnail row
  // and has no signal about why. Most common cause: matplotlib
  // not installed in the Python environment.
  if (evt.plot_diagnostic) {
    const note = document.createElement('div');
    note.className = 'tool-plot-diagnostic';
    note.textContent = evt.plot_diagnostic;
    body.appendChild(note);
  }

  // Inline plot thumbnails — every .png the script wrote into its
  // run dir (including those produced by `graph export` in Stata,
  // `ggsave` in R, `plt.savefig` in Python). These are the
  // RESEARCHER's view; the model only ever sees plots that came
  // through the manifest-allowlist gate in the runner.
  if (evt.plots && Array.isArray(evt.plots) && evt.plots.length > 0) {
    const grid = document.createElement('div');
    grid.className = 'tool-plots';
    evt.plots.forEach((plot) => {
      const tile = document.createElement('div');
      tile.className = 'tool-plot-tile';
      tile.title = plot.name;
      if (plot.data) {
        const img = document.createElement('img');
        img.alt = plot.name;
        img.src = `data:${plot.mime || 'image/png'};base64,${plot.data}`;
        img.addEventListener('click', () => showImageLightbox(img.src));
        tile.appendChild(img);
      } else {
        // Above the inline byte cap — render a placeholder with the
        // file size so the researcher knows it exists, plus an
        // "Open" button that hands off to the OS image viewer.
        const placeholder = document.createElement('div');
        placeholder.className = 'tool-plot-placeholder';
        placeholder.textContent = formatBytes(plot.size || 0);
        tile.appendChild(placeholder);
        if (plot.path && window.pywebview && window.pywebview.api &&
            typeof window.pywebview.api.open_path === 'function') {
          tile.style.cursor = 'pointer';
          tile.addEventListener('click', () => {
            window.pywebview.api.open_path(plot.path);
          });
        }
      }
      const caption = document.createElement('div');
      caption.className = 'tool-plot-caption';
      caption.textContent = plot.name;
      tile.appendChild(caption);
      grid.appendChild(tile);
    });
    body.appendChild(grid);
  }

  if (evt.run_dir) {
    const actions = document.createElement('div');
    actions.className = 'tool-actions';

    const lang = evt.language;  // "R" | "Stata" | "Python" | undefined
    const scriptFile =
      lang === 'Stata' ? 'script.do'
        : lang === 'Python' ? 'script.py'
        : 'script.R';
    const openInLabel =
      lang === 'Stata' ? 'Open in Stata'
        : lang === 'R' ? 'Open in R'
        : lang === 'Python' ? 'Open in Python'
        : 'Open script';
    const openMode =
      lang === 'Stata' ? 'run_stata'
        : lang === 'R' ? 'run_r'
        : lang === 'Python' ? 'run_python'
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
        : lang === 'Python'
        ? 'Open the Python script in your default .py handler.'
        : 'Open the script in its default app.';
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

function _parseToolResultPayload(evt) {
  if (!evt || !evt.text) return null;
  try {
    return JSON.parse(evt.text);
  } catch (_) {
    return null;
  }
}

function renderCanonicalResultTables(body, evt) {
  /* Append the per-result canonical tables to the tool-card body.
   *
   * Single-result run: render the one panel inline (it IS the
   * reading surface; nothing to compare against). Multi-result
   * run (N >= 2): collapse the panels into a ``<details>``
   * element closed by default, with a summary line naming the
   * count and id range. The model is now expected to call
   * ``compose_results`` and drop the comparison table into its
   * reply — that becomes the primary reading surface. The
   * collapsed panels stay accessible for audit (one click to
   * expand) without dominating the transcript with N stacked
   * tables.
   *
   * Same data source for both: each entry's ``markdown`` field,
   * canonical render from the sanitized payload. */
  const payload = _parseToolResultPayload(evt);
  const results = payload && Array.isArray(payload.results)
    ? payload.results
    : [];
  const rendered = results.filter((r) =>
    r && r.status === 'ok' && typeof r.markdown === 'string' && r.markdown.trim()
  );
  if (rendered.length === 0) return;

  const panel = document.createElement('div');
  panel.className = 'result-panel';

  function renderOneSection(r, idx) {
    const section = document.createElement('section');
    section.className = 'result-markdown';
    const header = document.createElement('div');
    header.className = 'result-header';
    header.textContent = r.label || r.result_id || ('Result ' + (idx + 1));
    section.appendChild(header);
    const tableWrap = document.createElement('div');
    tableWrap.className = 'result-markdown-body';
    if (window.NoraMarkdown) {
      tableWrap.innerHTML = window.NoraMarkdown.render(r.markdown);
    } else {
      tableWrap.textContent = r.markdown;
    }
    section.appendChild(tableWrap);
    return section;
  }

  if (rendered.length === 1) {
    panel.appendChild(renderOneSection(rendered[0], 0));
    body.appendChild(panel);
    return;
  }

  // Multi-result: collapsed by default. Summary line names what's
  // inside (count + id range) so the audit affordance is visible
  // even when collapsed; one click to expand for the per-result
  // detail. The composite table from ``compose_results`` lives in
  // the model's reply, not here.
  const details = document.createElement('details');
  details.className = 'result-panel-collapsed';
  const summary = document.createElement('summary');
  summary.className = 'result-panel-summary';
  const firstId = rendered[0].result_id || '?';
  const lastId = rendered[rendered.length - 1].result_id || '?';
  const idRange = firstId === lastId
    ? firstId
    : `${firstId}–${lastId}`;
  summary.textContent =
    `${rendered.length} regressions stored: ${idRange} (click to expand)`;
  details.appendChild(summary);
  rendered.forEach((r, idx) => {
    details.appendChild(renderOneSection(r, idx));
  });
  panel.appendChild(details);
  body.appendChild(panel);
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

// "Scroll to latest" floating button. Anchored above the composer
// in the markup; visibility tracks whether the transcript is near
// its bottom. The threshold accommodates the natural overshoot of
// in-flight typewriter renders without flashing the button.
const scrollToBottomBtn = document.getElementById('scroll-to-bottom');
const SCROLL_TO_BOTTOM_THRESHOLD = 100;

function updateScrollToBottomVisibility() {
  if (!scrollToBottomBtn || !messagesEl) return;
  const distanceFromBottom = (
    messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight
  );
  const nearBottom = distanceFromBottom < SCROLL_TO_BOTTOM_THRESHOLD;
  scrollToBottomBtn.classList.toggle('hidden', nearBottom);
}

if (messagesEl) {
  messagesEl.addEventListener('scroll', updateScrollToBottomVisibility);
  // Container resizes on sidebar drag, file panel toggle, etc. —
  // the "near bottom" check depends on clientHeight, so we re-poll
  // on resize too.
  window.addEventListener('resize', updateScrollToBottomVisibility);
}

if (scrollToBottomBtn) {
  scrollToBottomBtn.addEventListener('click', () => {
    // Smooth scroll matches user expectation that this is a
    // navigation gesture, not a snap. ``scrollToBottom()`` (the
    // existing helper) is left as the immediate snap that other
    // call sites use after appending content.
    if (typeof messagesEl.scrollTo === 'function') {
      messagesEl.scrollTo({
        top: messagesEl.scrollHeight,
        behavior: 'smooth',
      });
    } else {
      scrollToBottom();
    }
  });
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

function updateContextChip(occupiedTokens) {
  /* Updates the "Context X / Y (Z%)" chip below the composer.
   * ``occupiedTokens`` is "context occupied after this turn" —
   * the prompt this turn loaded plus the response just produced
   * (see the case 'turn_done' handler for the breakdown).
   *
   * Provider semantics:
   * - Anthropic: input_tokens + cache_read + cache_creation
   *   together span the whole prompt-side window via the prompt
   *   cache; output_tokens adds the just-produced reply.
   * - OpenAI: with previous_response_id, the SDK reports
   *   input_tokens for the FULL prompt at each round (cached
   *   prefix included). The provider yields the LAST round's
   *   value (= peak prompt size for the turn); cache fields stay
   *   empty on this path because OpenAI's cached_tokens is a
   *   subset of input_tokens, not additive.
   *
   * Unhides the chip on first update and scales the ceiling up
   * if the observed usage exceeds the default 200k window
   * (Opus 4.7 1M variant, Sonnet 4.6, etc.) so the ratio stays
   * meaningful. */
  if (!contextChip) return;
  if (typeof occupiedTokens !== 'number' || occupiedTokens < 0) return;

  // Session-monotonic clamp. Per-turn usage events report whatever
  // the most recent request measured — for OpenAI that's the last
  // Responses round's input_tokens (+ output); for Anthropic it's
  // the prompt-cache snapshot for that turn. Both can DECREASE
  // turn-to-turn (a tool-heavy turn reports a peak prompt; a plain
  // follow-up reports a smaller one) even though the underlying
  // chain only grows. Take the max so the chip tracks the chain,
  // not the last measurement.
  occupiedTokens = Math.max(occupiedTokens, sessionContextHighWater);
  sessionContextHighWater = occupiedTokens;
  // Persist per-cwd so a re-ready (model swap, reconnect) or a
  // session-switch round-trip restores the watermark instead of
  // dropping to zero.
  if (currentCwd) sessionHighWaters.set(currentCwd, occupiedTokens);

  // Trust the authoritative ceiling set in updateModelChip (line ~3085)
  // from the model-info payload. The previous code auto-scaled the
  // ceiling to 2M whenever usage exceeded 1M, which silently
  // misreported overruns: an Opus 4.7 1M turn at 1426k chars showed
  // "71%" of 2M instead of the truthful "143% of 1M, you've blown
  // past the window". Anthropic has no 2M model today, so the
  // heuristic never matched reality. If the chip looks pinned at
  // 100% on a different model in the future, fix the model registry
  // (catalog.py / updateModelChip), not the chip.

  const fmt = (n) => (n >= 10_000 ? (n / 1000).toFixed(1) + 'k' : n.toString());
  const ceilingLabel = contextWindow >= 1_000_000
    ? (contextWindow / 1_000_000) + 'M'
    : (contextWindow / 1000) + 'k';
  const rawPct = Math.round((occupiedTokens / contextWindow) * 100);
  // Show the real ratio uncapped so an over-window state is visible
  // ("143%") instead of clamped to "100%" and blending into a normal
  // full-but-fine state.
  contextChip.textContent = `Context ${fmt(occupiedTokens)} / ${ceilingLabel} (${rawPct}%)`;
  contextChip.classList.remove('hidden');

  // Visual warning as context fills up — dim at low use, warm as it
  // climbs, red near the ceiling, the strongest tone past 100% so a
  // window overrun is visually unmistakable.
  contextChip.classList.remove('warn', 'danger', 'over');
  if (rawPct >= 100) contextChip.classList.add('over');
  else if (rawPct >= 90) contextChip.classList.add('danger');
  else if (rawPct >= 70) contextChip.classList.add('warn');
}

function updatePolicyChip(policy) {
  /* Refresh the compact chip label + the per-dataset dropdowns
   * inside the popup. Called on session start and after any policy
   * mutation. The topbar Files chip is refreshed from a separate
   * ``list_session_files`` endpoint that includes scripts and
   * graphs too — the policy payload covers data files only because
   * the SDC layer's permission tiers are dataset-specific. */
  if (!policy || !policy.datasets || policy.datasets.length === 0) {
    policyChip.classList.add('hidden');
  } else {
    policyChip.classList.remove('hidden');
    policyChipLabel.textContent = compactPolicyLabel(policy);
    policyPopupBuiltFor = policy;
    policyPopup.innerHTML = '';
    policyPopup.appendChild(buildPolicyPopup(policy));
  }
  refreshFilesChip();
}

// ----- topbar files chip ------------------------------------------------
//
// "Files" lives in the topbar's centered cluster, next to the
// session-title pill. Read-only — permission-tier editing stays in
// the bottom Permission chip. The chip exists so a researcher can
// answer "did my upload land?" at a glance, including scripts and
// graphs that don't show up in the data-only Permission panel.

const filesChip = document.getElementById('files-chip');
const filesChipLabel = document.getElementById('files-chip-label');
const filesPopup = document.getElementById('files-popup');

const FILES_KIND_LABELS = {
  data: 'Data',
  script: 'Scripts',
  graph: 'Graphs',
  log: 'Logs',
};

async function refreshFilesChip() {
  /* Pull the full session file list from the bridge and re-render
   * the chip + popup. Called from updatePolicyChip and from any
   * other event that might have changed cwd contents (drag-drop,
   * dialog upload, session switch).
   *
   * Note: Data files are filtered out of this popup — they're
   * already shown in the bottom Permission chip with their depth
   * tier. The Files popup is the surface for the OTHER session
   * artifacts (scripts, graphs, logs) that have no home elsewhere.
   */
  // The Files panel and the @-mention dropdown share the same
  // source of truth (session-resident files). Whenever this chip
  // refreshes, bust the mention cache so the next "@" pull
  // re-fetches.
  invalidateMentionCache();
  if (!filesChip) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.list_session_files !== 'function') return;
  let res;
  try {
    res = await window.pywebview.api.list_session_files();
  } catch (err) {
    console.warn('list_session_files failed', err);
    return;
  }
  const allFiles = (res && res.files) || [];
  // Drop the data group — it already shows in the Permission chip
  // alongside each file's schema-depth tier. Showing the same names
  // twice (once here, once there) just creates two surfaces that
  // can drift.
  const files = allFiles.filter((f) => f.kind !== 'data');
  if (files.length === 0) {
    filesChip.classList.add('hidden');
    if (filesPopup) filesPopup.classList.add('hidden');
    filesChip.classList.remove('open');
    return;
  }
  filesChip.classList.remove('hidden');
  if (filesChipLabel) {
    filesChipLabel.textContent = `Files · ${files.length}`;
  }
  if (filesPopup) {
    filesPopup.innerHTML = '';
    const wrap = document.createElement('div');
    const header = document.createElement('div');
    header.className = 'policy-popup-header';
    header.innerHTML =
      '<strong>Files</strong>. Scripts, graphs, and logs uploaded to '
      + 'this session. Click a script to attach its content to your '
      + 'next message. Data files are listed in the Permission chip.';
    wrap.appendChild(header);
    // Group by kind so scripts / graphs / logs land in their own
    // sections — same shape the model picker uses for providers.
    const byKind = new Map();
    files.forEach((f) => {
      const k = f.kind || 'script';
      if (!byKind.has(k)) byKind.set(k, []);
      byKind.get(k).push(f);
    });
    // Render order: graphs first (the visual outputs the
    // researcher iterates on), then scripts (sometimes attached
    // mid-chat), then logs (rarely interacted with). Previously
    // scripts came first, which buried plots below text-only
    // rows even though plots are the most-clicked kind.
    ['graph', 'script', 'log'].forEach((kind) => {
      const rows = byKind.get(kind);
      if (!rows || rows.length === 0) return;
      const groupHeader = document.createElement('div');
      groupHeader.className = 'model-group-header';
      groupHeader.textContent = FILES_KIND_LABELS[kind] || kind;
      wrap.appendChild(groupHeader);
      rows.forEach((f) => {
        wrap.appendChild(buildFilesRow(kind, f));
      });
    });
    filesPopup.appendChild(wrap);
  }
}

function buildFilesRow(kind, f) {
  /* Build one row in the Files popup. Layout: [primary action]
   * [title / thumbnail] [delete]. The primary action varies by
   * kind:
   *   - graph (image with thumbnail data): copy-to-clipboard
   *   - graph (no thumbnail data, e.g. .gph): open externally
   *   - script: send (attach to next message)
   *   - log: no primary action
   * Delete is always on the right.
   */
  const row = document.createElement('div');
  row.className = 'files-row files-row-actionable';
  row.dataset.kind = kind;
  row.dataset.path = f.path || '';
  row.title = f.path || f.name;

  // Left action (kind-specific).
  const leftAction = document.createElement('button');
  leftAction.type = 'button';
  leftAction.className = 'files-row-action files-row-action-left';
  let primaryConfigured = false;
  if (kind === 'graph' && f.data) {
    leftAction.title = 'Copy image to clipboard';
    leftAction.setAttribute('aria-label', 'Copy image');
    leftAction.innerHTML = COPY_ICON_SVG;
    leftAction.addEventListener('click', (ev) => {
      ev.stopPropagation();
      copyImageToClipboard(f.data, f.mime || 'image/png', f.name);
    });
    primaryConfigured = true;
  } else if (kind === 'graph' && f.path) {
    leftAction.title = 'Open in default viewer';
    leftAction.setAttribute('aria-label', 'Open');
    leftAction.innerHTML = OPEN_ICON_SVG;
    leftAction.addEventListener('click', (ev) => {
      ev.stopPropagation();
      if (window.pywebview && window.pywebview.api &&
          typeof window.pywebview.api.open_path === 'function') {
        window.pywebview.api.open_path(f.path);
      }
    });
    primaryConfigured = true;
  } else if (kind === 'script') {
    leftAction.title = 'Send to next message';
    leftAction.setAttribute('aria-label', 'Attach to next message');
    leftAction.innerHTML = SEND_ICON_SVG;
    leftAction.addEventListener('click', (ev) => {
      ev.stopPropagation();
      attachSessionFile(f.name);
    });
    primaryConfigured = true;
  }
  if (primaryConfigured) {
    row.appendChild(leftAction);
  } else {
    // Spacer keeps the title aligned with rows that DO have a
    // left action — visual alignment beats squeezing an extra
    // pixel of horizontal space.
    const spacer = document.createElement('span');
    spacer.className = 'files-row-action-spacer';
    row.appendChild(spacer);
  }

  // Center: title + (for image rows) the thumbnail.
  const center = document.createElement('div');
  center.className = 'files-row-center';
  if (kind === 'graph' && f.data) {
    const thumb = document.createElement('img');
    thumb.className = 'files-row-thumb';
    thumb.alt = f.name;
    thumb.src = `data:${f.mime || 'image/png'};base64,${f.data}`;
    thumb.addEventListener('click', (ev) => {
      ev.stopPropagation();
      showImageLightbox(thumb.src);
    });
    center.appendChild(thumb);
  }
  const caption = document.createElement('div');
  caption.className = 'files-row-caption';
  caption.textContent = f.name;
  center.appendChild(caption);
  row.appendChild(center);

  // Right action: delete. Always present so every file in the
  // panel can be removed with one click. Matches the session-list
  // delete affordance — a plain ``×`` glyph rather than an icon —
  // so the visual vocabulary stays consistent across delete
  // surfaces.
  const deleteBtn = document.createElement('button');
  deleteBtn.type = 'button';
  deleteBtn.className = 'files-row-action files-row-action-right files-row-action-danger';
  deleteBtn.title = 'Delete file';
  deleteBtn.setAttribute('aria-label', 'Delete file');
  deleteBtn.textContent = '×';
  deleteBtn.addEventListener('click', (ev) => {
    ev.stopPropagation();
    deleteSessionFile(f.path || f.name, f.name);
  });
  row.appendChild(deleteBtn);
  return row;
}


// Inline SVG icons. Small, monochrome — color is set via CSS so
// the icon picks up the row's hover/focus colors.
const COPY_ICON_SVG = (
  '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">' +
  '<path fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" ' +
  'd="M5 5V2.5A.5.5 0 0 1 5.5 2h7a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5H10' +
  'M3.5 5h7a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5h-7a.5.5 0 0 1-.5-.5v-9a.5.5 0 0 1 .5-.5z"/>' +
  '</svg>'
);

const SEND_ICON_SVG = (
  '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">' +
  '<path fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" ' +
  'd="M2 8L14 2l-3 12-3-5-6-1z"/>' +
  '</svg>'
);

const OPEN_ICON_SVG = (
  '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">' +
  '<path fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" ' +
  'd="M9 2h5v5M14 2L7 9M3 4h3M3 4v9h9v-3"/>' +
  '</svg>'
);

async function copyImageToClipboard(base64Data, mime, name) {
  /* Copy a thumbnail image to the system clipboard via the
   * Clipboard API. WKWebView supports ClipboardItem; if for any
   * reason it doesn't (older macOS), we fall back to a toast
   * pointing at "Show folder" so the researcher isn't stuck.
   */
  try {
    if (!navigator.clipboard || typeof ClipboardItem === 'undefined') {
      toast('Clipboard API unavailable in this WebView; use Show folder.', 'info');
      return;
    }
    const binary = atob(base64Data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const blob = new Blob([bytes], { type: mime || 'image/png' });
    await navigator.clipboard.write([
      new ClipboardItem({ [blob.type]: blob }),
    ]);
    toast('Copied ' + (name || 'image') + ' to clipboard.', 'success');
  } catch (err) {
    console.warn('copyImageToClipboard failed', err);
    toast('Copy failed: ' + (err && err.message ? err.message : err), 'error');
  }
}


async function deleteSessionFile(path, displayName) {
  /* Delete a file via the bridge. Confirms first because
   * unlinking is irreversible and the Files panel doesn't have
   * an undo. Refreshes the panel + composer chips on success
   * so the row vanishes immediately.
   */
  if (!path) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.delete_session_file !== 'function') {
    toast('Restart Nora to enable file deletion.', 'info');
    return;
  }
  const label = displayName || path;
  const ok = window.confirm(`Delete ${label}?\n\nThis cannot be undone.`);
  if (!ok) return;
  try {
    const res = await window.pywebview.api.delete_session_file(path);
    if (!res || !res.ok) {
      const reason = (res && res.reason) || 'unknown';
      toast('Could not delete: ' + reason, 'error');
      return;
    }
    toast('Deleted ' + (res.name || label) + '.', 'success');
    refreshFilesChip();
    // Composer chips may have referenced the file too; re-render.
    renderAttachments();
  } catch (err) {
    console.warn('delete_session_file failed', err);
    toast('Could not delete: ' + (err && err.message ? err.message : err), 'error');
  }
}


async function attachSessionFile(name) {
  /* Stage a session-resident script as inline context for the next
   * message. The bridge writes into ``_pending_script_attachments``
   * (same list drag-drop populates) so the message-prefix builder
   * picks it up automatically on send. JS-side, we mirror what the
   * drag-drop path does: push the name onto ``stagedDataNotices``
   * so the composer chip shows up too. */
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.attach_session_file !== 'function') return;
  try {
    const res = await window.pywebview.api.attach_session_file(name);
    if (!res || !res.ok) {
      const reason = (res && res.reason) || 'unknown';
      toast('Could not attach: ' + reason, 'error');
      return;
    }
    if (res.already_attached) {
      toast(name + ' is already attached to your next message.', 'info');
    } else {
      // Mirror the drag-drop chip so the researcher sees the same
      // visual confirmation in both flows.
      if (addStagedDataNotices([res.name || name])) renderAttachments();
      toast(name + ' attached to your next message.', 'success');
    }
    // Close the popup so the composer becomes the obvious next
    // surface to interact with.
    if (filesPopup) filesPopup.classList.add('hidden');
    if (filesChip) filesChip.classList.remove('open');
  } catch (err) {
    console.warn('attach_session_file failed', err);
    toast('Could not attach: ' + err, 'error');
  }
}

if (filesChip && filesPopup) {
  filesChip.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOpen = !filesPopup.classList.contains('hidden');
    filesPopup.classList.toggle('hidden');
    filesChip.classList.toggle('open', !isOpen);
  });
  document.addEventListener('click', (e) => {
    if (filesPopup.classList.contains('hidden')) return;
    if (filesPopup.contains(e.target) || filesChip.contains(e.target)) return;
    filesPopup.classList.add('hidden');
    filesChip.classList.remove('open');
  });
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
  // "default: …" crutch. The active tier is visible in the row
  // selection itself.
  header.innerHTML =
    '<strong>Permission</strong>: ceiling on variable details ' +
    'Nora sees per dataset. It can ask for less, never more.';
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
  /* Fetch the model catalog from the backend and render the popup.
   *
   * The backend's ``current`` is authoritative — it reflects the
   * per-session choice restored from ``.nora/session_state.json`` if
   * the researcher already used a particular model in this session,
   * or the global default otherwise. We used to override it with a
   * localStorage value here, but that turned a per-session feature
   * into a global one and silently swapped models on session open.
   */
  if (!modelChip || !window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.list_models !== 'function') return;
  try {
    const res = await window.pywebview.api.list_models();
    if (!res || !res.ok) return;
    availableModels = res.models || [];
    currentModelId = res.current;
    renderModelChip();
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

function renderModelPopup() {
  if (!modelPopup) return;
  modelPopup.innerHTML = '';
  const wrap = document.createElement('div');
  const header = document.createElement('div');
  header.className = 'policy-popup-header';
  header.innerHTML = '<strong>Model</strong>. The active model for this session.';
  wrap.appendChild(header);

  // Group models by provider so the picker reads as
  //   Anthropic
  //     Sonnet 4.6 (1M)
  //     Opus 4.7 (1M)
  //     Haiku 4.5
  //   OpenAI
  //     GPT-5
  //     GPT-5 mini
  // Models for un-authed providers stay in the list but render
  // disabled with a "Configure auth" hint so the researcher can see
  // the option exists without being able to silently pick it.
  const byProvider = new Map();
  availableModels.forEach((m) => {
    const p = m.provider || 'anthropic';
    if (!byProvider.has(p)) byProvider.set(p, []);
    byProvider.get(p).push(m);
  });

  const providerOrder = ['anthropic', 'openai'];
  providerOrder.forEach((p) => {
    const models = byProvider.get(p);
    if (!models || models.length === 0) return;
    const groupHeader = document.createElement('div');
    groupHeader.className = 'model-group-header';
    groupHeader.textContent = p === 'anthropic' ? 'Anthropic' : 'OpenAI';
    wrap.appendChild(groupHeader);

    models.forEach((m) => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'model-option';
      if (m.id === currentModelId) row.classList.add('active');
      const available = m.available !== false;
      if (!available) {
        row.classList.add('unconfigured');
      }

      // Hover tooltip — only for the auth state. Researchers can
      // open the per-row pricing link for live pricing detail rather
      // than relying on canned text that goes stale.
      if (!available) {
        row.dataset.tooltip = 'Click to configure';
      } else {
        delete row.dataset.tooltip;
      }

      const name = document.createElement('span');
      name.className = 'model-option-name';
      name.textContent = m.label;
      row.appendChild(name);

      // Right-side cluster: context window + per-row pricing link.
      // Wrapped so the link sits inside the picker row's hover area
      // but stops click propagation so it doesn't trigger a model
      // switch on the way out.
      const right = document.createElement('span');
      right.className = 'model-option-right';

      const ctx = document.createElement('span');
      ctx.className = 'model-option-ctx';
      ctx.textContent = formatContextWindow(m.context_window);
      right.appendChild(ctx);

      if (m.pricing_url) {
        const priceLink = document.createElement('button');
        priceLink.type = 'button';
        priceLink.className = 'model-option-price';
        priceLink.title = 'View pricing';
        priceLink.textContent = '$';
        priceLink.addEventListener('click', (e) => {
          e.stopPropagation();
          openExternal(m.pricing_url);
        });
        right.appendChild(priceLink);
      }
      row.appendChild(right);

      row.addEventListener('click', () => {
        modelPopup.classList.add('hidden');
        modelChip.classList.remove('open');
        if (available) {
          setModel(m.id, /*silent=*/false);
        } else {
          // Clicking an un-authed model jumps to the auth screen so
          // the researcher can paste a key for that provider without
          // having to bounce through landing first.
          openAuthScreen();
        }
      });
      wrap.appendChild(row);
    });
  });

  modelPopup.appendChild(wrap);
}

function formatContextWindow(n) {
  if (!n) return '';
  if (n >= 1_000_000) {
    // 1.05M reads better than 1.0500000M; one decimal place is plenty.
    const m = n / 1_000_000;
    return (Math.round(m * 100) / 100) + 'M ctx';
  }
  return (n / 1000) + 'k ctx';
}

function showImageLightbox(url) {
  /* Open an in-page overlay showing ``url`` at full resolution.
   * Used for image attachments — the bridge's ``open_external`` is
   * allowlisted to specific HTTPS pricing pages and won't accept
   * blob: URLs anyway. Click anywhere or press Esc to close. */
  if (!url) return;
  const overlay = document.createElement('div');
  overlay.className = 'image-lightbox';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-label', 'Enlarged image');
  overlay.tabIndex = -1;
  const big = document.createElement('img');
  big.src = url;
  big.alt = 'Enlarged image';
  big.className = 'image-lightbox-img';
  overlay.appendChild(big);
  const close = () => {
    overlay.remove();
    document.removeEventListener('keydown', onKey);
  };
  const onKey = (e) => {
    if (e.key === 'Escape') close();
  };
  overlay.addEventListener('click', close);
  document.addEventListener('keydown', onKey);
  document.body.appendChild(overlay);
  // Focus so Esc works without an extra click first.
  overlay.focus();
}

async function openExternal(url) {
  /* Hand a URL to the OS default browser via the bridge. The bridge
   * allowlists URLs against the known pricing pages so this can't be
   * coerced into navigating to attacker-controlled sites by a stray
   * tool result. */
  if (!url) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.open_external !== 'function') return;
  try {
    const res = await window.pywebview.api.open_external(url);
    if (res && !res.ok) {
      console.warn('open_external rejected:', res.reason);
    }
  } catch (err) {
    console.warn('open_external failed', err);
  }
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
    // ``data-path`` is read by setSessionBusy() to flip the busy
    // dot on this row when a turn starts/ends — including for
    // background sessions the user isn't currently looking at.
    btn.dataset.path = s.path;
    if (busySessions.has(s.path)) btn.classList.add('busy');
    const when = document.createElement('div');
    when.className = 'session-when';
    // Busy dot lives INSIDE the title line so it pulses right
    // next to the timestamp instead of off in the corner. CSS
    // hides it by default; ``.busy`` on the parent reveals it.
    const dot = document.createElement('span');
    dot.className = 'session-busy-dot';
    dot.setAttribute('aria-hidden', 'true');
    when.appendChild(dot);
    const whenText = document.createElement('span');
    whenText.className = 'session-when-text';
    whenText.textContent = formatSessionWhen(s.timestamp);
    when.appendChild(whenText);
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
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.delete_session !== 'function') {
    toast('Restart Nora to enable session deletion.', 'error');
    return;
  }
  const when = formatSessionWhen(s.timestamp);
  const sizeText = typeof s.size === 'number' ? ' (' + formatBytes(s.size) + ')' : '';
  // Active-session deletes get a more explicit prompt — the
  // researcher is wiping the chat they're currently looking at,
  // not a stale one in the sidebar. The default prompt covers
  // the data-loss content; the prefix makes the "you're in this
  // one right now" angle unambiguous.
  const headline = isCurrent
    ? `Delete the session you're currently in (from ${when})${sizeText}?`
    : `Delete session from ${when}${sizeText}?`;
  const ok = window.confirm(
    `${headline}\n\nThis removes the data copies, run logs, results.db, and chat history. Cannot be undone.`
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
    // Active-session delete: bridge dropped self.cwd; the page
    // must navigate to the landing screen to match. Without this
    // the chat surface still shows the (now stale) transcript and
    // the next send_message would crash on a missing cwd.
    if (res.was_active) {
      showLanding();
    } else {
      loadSessions();
    }
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
      const addedNotices = addStagedDataNotices(added);
      if (images.length > 0 || addedNotices) renderAttachments();

      // Summary toast: describe what happened in one line.
      const parts = [];
      if (added.length === 1) parts.push('Added ' + added[0]);
      else if (added.length > 1) parts.push('Added ' + added.length + ' files');
      if (images.length === 1) parts.push('attached 1 image');
      else if (images.length > 1) parts.push('attached ' + images.length + ' images');
      if (parts.length > 0) {
        toast(parts.join(' · '), 'success');
      }
      if (skipped.length > 0) {
        toast('Skipped: ' + skipped.join(', '), 'info');
      }
      const skippedExisting = res.skipped_existing || [];
      if (skippedExisting.length > 0) {
        // Distinct toast colour so the researcher sees this is "we
        // didn't overwrite", not "we couldn't read these".
        toast(
          'Already in this session, not overwritten: '
          + skippedExisting.join(', '),
          'info'
        );
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
  // Three-stage state machine on startup:
  //   needs_auth     → researcher hasn't configured any provider yet.
  //                    Show the auth screen first so the chat can't
  //                    silently fail with "no API key" later.
  //   needs_session  → auth is good but no working directory chosen.
  //                    Land on the drop / choose-files screen.
  //   ready          → both done; jump into chat.
  // Any exception from ui_ready falls back to showLanding() — better
  // to be too permissive than to wedge the page on a hard error.
  try {
    const state = await window.pywebview.api.ui_ready();
    if (state && state.state === 'needs_auth') {
      showAuth(state.auth);
    } else if (state && state.state === 'ready') {
      showChat(state);
    } else {
      showLanding();
    }
  } catch (err) {
    console.error('ui_ready failed', err);
    showLanding();
  }
});
