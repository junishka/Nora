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
const authContinueHint = document.getElementById('auth-continue-hint');
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

// Topbar pill — click to reveal the session folder in Finder. The
// pill always shows the abbreviated cwd path (formatCwd) so the
// researcher can see at a glance WHERE on disk Nora is writing
// scripts, logs, generated data, and plots; clicking opens that
// folder so they can inspect outputs in the OS file manager
// directly. Renaming a session lives only on the sidebar's pencil
// button — the topbar used to double as a rename trigger, but
// because it shared display state with the auto-derived title,
// renames flowed one way (topbar→sidebar) and not the other
// (sidebar→topbar), which was confusing. Splitting the two
// concerns — sidebar = name, topbar = path — fixes the asymmetry.
if (cwdEl) {
  cwdEl.title = 'Click to open this session\'s folder in Finder';
  cwdEl.setAttribute('role', 'button');
  cwdEl.setAttribute('tabindex', '0');
  const openSessionFolder = async () => {
    if (!currentCwd) return;
    if (!window.pywebview || !window.pywebview.api) return;
    if (typeof window.pywebview.api.open_path !== 'function') {
      toast('Restart Nora to enable folder reveal.', 'info');
      return;
    }
    try {
      const res = await window.pywebview.api.open_path(currentCwd);
      if (!res || !res.ok) {
        const reason = (res && res.reason) || 'unknown';
        toast('Could not open folder: ' + reason, 'error');
      }
    } catch (err) {
      console.warn('open_path failed', err);
      toast(
        'Could not open folder: '
        + (err && err.message ? err.message : err),
        'error',
      );
    }
  };
  cwdEl.addEventListener('click', openSessionFolder);
  cwdEl.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    openSessionFolder();
  });
}

// Context-window ceiling for the chip's ratio display. Updated
// whenever the researcher picks a model (see updateModelChip) from
// the catalog's per-model ``context_window`` — every current entry
// is 1M (Claude 5 family) or 1.05M (GPT-5.6 family). The starting
// 1M matches the default model (Sonnet 5).
const DEFAULT_CONTEXT_WINDOW = 1_000_000;
let contextWindow = DEFAULT_CONTEXT_WINDOW;
// Last solid count returned by ``count_next_context``. The chip
// keeps showing this until a new count lands — no estimates, no
// projections from pending messages, no provider ``turn_done``
// echoes. Triggers (turn complete, rewind success, session
// open/switch, attachment add/remove) call ``triggerContextRecount``
// which fades the chip and updates this when the backend returns.
let lastContextCount = null;     // {tokens, exact, ceiling}
let contextCountRequestId = 0;   // monotonic counter for stale-response rejection

// (No high-water clamp on the context chip. Earlier we Math.max'd
// each turn_done's reported usage against a per-cwd watermark so the
// chip "only grew." That hid measurement asymmetries between
// providers — Anthropic's MAX-across-rounds inflation got pinned in
// place forever, and OpenAI's silent truncation never visibly shrank
// the chip even though it was happening. The chip now reflects
// exactly what the last completed turn reported. The conversation
// chain grows monotonically on its own, so under normal use the
// chip will only rise. If it drops, that's a real signal — a
// reconnect that lost in-context state, a model swap that opened a
// fresh session, a truncation event — and surfacing it is the
// point.)

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

// Sessions whose in-flight turn the researcher just hit Stop on.
// Cancellation is async — the bridge has to propagate it through the
// asyncio task and the provider stream, which can leave tokens or
// tool calls already in transit. Without this set, those latent
// events keep painting into the transcript even though the
// researcher signalled "stop". We add the cwd on Stop click and
// remove it on the next terminal event for that session OR on the
// next user send (whichever comes first, in case a terminal never
// arrives because the SDK closed without one). While a cwd is in
// the set, ``nora_event`` drops non-terminal events for that
// session.
// Cancelled-turn drop list. The runner stamps every event with a
// per-turn id; when Stop fires we add the in-flight id here and
// keep dropping any late events that carry it. Crucially, this set
// is NEVER cleared when a new message starts — that was the prior
// bug: ``cancelledCwds.delete(currentCwd)`` on send let late events
// from the previous (cancelled) turn slip through after the new
// turn started, because the suppression key was a cwd rather than
// a turn id. Each new turn gets a fresh id, so the new turn's
// events naturally pass the filter without anyone having to clear
// state. The backend dispatcher applies the same drop authoritatively
// (see ``_dispatch_event`` in ui.py) — this set is best-effort
// defense in depth.
//
// Bounded growth: oldest ids are evicted past ~CANCELLED_TURN_ID_CAP
// entries. After eviction a stray late event can render, but by
// then the turn is far enough back in history that a brief flash
// before the staleness sweep is acceptable.
const CANCELLED_TURN_ID_CAP = 256;
const cancelledTurnIds = new Set();
const cancelledTurnIdOrder = [];

function markTurnCancelled(turnId) {
  if (!turnId) return;
  if (cancelledTurnIds.has(turnId)) return;
  cancelledTurnIds.add(turnId);
  cancelledTurnIdOrder.push(turnId);
  while (cancelledTurnIdOrder.length > CANCELLED_TURN_ID_CAP) {
    const evicted = cancelledTurnIdOrder.shift();
    cancelledTurnIds.delete(evicted);
  }
}

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

// Display labels for provider ids. The auth bridge speaks lowercase
// ids ("anthropic", "openai"); UI copy should never leak those raw.
// Falls back to capitalising the id if a new provider is added before
// this map catches up.
const PROVIDER_LABELS = {
  anthropic: 'Anthropic',
  openai: 'OpenAI',
};
function providerLabel(id) {
  if (!id) return '';
  if (PROVIDER_LABELS[id]) return PROVIDER_LABELS[id];
  return id.charAt(0).toUpperCase() + id.slice(1);
}

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
  // Drop focus into the first unconfigured input so a first-launch
  // researcher can paste straight away without tabbing to find the
  // field. If every provider is already authed (e.g., they hit
  // "Manage providers" from landing just to peek), skip — stealing
  // focus to a field they don't need to touch is noisy.
  const firstEmpty = authEl.querySelector(
    '.auth-provider:not(.configured) [data-role="key-input"]'
  );
  if (firstEmpty) {
    // requestAnimationFrame so focus lands after the panel has
    // un-hidden; focusing a still-hidden element is a no-op.
    requestAnimationFrame(() => firstEmpty.focus());
  }
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
        // Tri-state. ``keyring_unavailable`` is distinct from ``missing``:
        // a locked or denied macOS Keychain prompt previously rendered
        // identically to "no credential here", so a researcher might
        // re-paste a key they already had (or dismiss the prompt
        // thinking the credential was gone when it was still stored).
        // Show a different message so they retry the prompt instead.
        statusEl.textContent = info.status === 'keyring_unavailable'
          ? 'Keychain unavailable — could not check'
          : 'Not configured';
      }
    }
    if (forgetBtn) {
      forgetBtn.disabled = !info.has_keyring_entry;
    }
  });
  const anyAuthed = !!(authPayload && authPayload.any_authed);
  if (authContinueBtn) {
    authContinueBtn.disabled = !anyAuthed;
  }
  if (authContinueHint) {
    // Swap between the "why is Continue gray?" prompt and a quiet
    // ready-state confirmation. Keeping the element rendered (rather
    // than show/hide) prevents the footer from jumping vertically
    // when the researcher's first Save flips the state.
    authContinueHint.textContent = anyAuthed
      ? 'Ready when you are.'
      : 'Save at least one provider to continue.';
    authContinueHint.classList.toggle('ready', anyAuthed);
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
            setAuthStatus(`${providerLabel(provider)} key saved.`, 'ok');
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
            setAuthStatus(`${providerLabel(provider)} credential removed.`, 'ok');
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

// Inline "create an API key" links inside each provider's help text.
// WKWebView would otherwise navigate the whole webview to the provider
// console, blowing away the auth state. Intercept the click and hand
// the URL to the OS browser via the allowlisted bridge.
if (authEl) {
  authEl.addEventListener('click', (e) => {
    const link = e.target.closest('.auth-help-link');
    if (!link) return;
    e.preventDefault();
    const url = link.getAttribute('data-external-url');
    if (url) openExternal(url);
  });
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

      // Title = the resolved session title (custom name if set, else
      // the dataset filename, else "<first> +N"). Renders single-line
      // with the relative-age label pinned to the right.
      const title = document.createElement('span');
      title.className = 'landing-session-title';
      title.textContent = s.title || (s.datasets.length
        ? s.datasets.join(', ')
        : '(no data files)');
      btn.appendChild(title);

      const age = document.createElement('span');
      age.className = 'landing-session-age';
      age.textContent = formatSessionAge(s.timestamp);
      btn.appendChild(age);

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

  // Topbar shows the abbreviated session path (``~/.nora-sessions/
  // <id>``) as a clickable handle to "reveal in Finder". The full
  // path stays in the title-attribute tooltip; the pill's click
  // handler hands the path to ``open_path`` so the researcher can
  // inspect Nora's scripts / logs / generated data on disk. The
  // session's friendly name (custom_name or auto-derived) lives on
  // the sidebar row, not here — see the cwdEl click-handler comment
  // above for the rationale.
  cwdEl.textContent = formatCwd(payload.cwd || '');
  cwdEl.title = payload.cwd
    ? payload.cwd + '. Click to open in Finder.'
    : '';

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
  // Hide the chip until the first turn_done arrives — without a
  // measurement we have nothing honest to display. Reset the
  // last-rendered count too so a fresh session doesn't re-render
  // the previous session's number against the new model's window.
  // Drop the previous session's count so we don't briefly render
  // it against the new session's eventual count. ``replayHistory``
  // triggers a recount once the new history loads.
  lastContextCount = null;
  if (contextChip) {
    contextChip.classList.add('hidden');
    contextChip.classList.remove('stale');
  }

  // Pass the cwd we just committed to so replay can abandon if the
  // user has already switched again by the time the history fetch
  // resolves. Without the guard, two quick A→B switches race the
  // get_chat_history responses and a late-arriving A history paints
  // into B's transcript.
  replayHistory(currentCwd);
  rotatePlaceholder();
  input.focus();
}

// Turns that yielded no visible reply artifacts. Keep them around
// just long enough for the researcher to notice the failure, then
// sweep them the next time a new message is sent so the transcript
// doesn't fill with dead-end bubbles.
// Live turn per session, keyed by cwd — NOT a single global.
//
// This used to be one ``activeLiveTurn`` object meaning "the focused
// session's in-flight turn", written by whichever session sent last.
// Focus changes never rebound it, so with two sessions running it
// pointed at the wrong turn: start A, switch to B and start B, switch
// back to A, press Stop. The bridge correctly cancelled focused A,
// but the JS blacklisted the stale global — B's id — and every
// subsequent B event including its terminal one was dropped at the
// top of ``nora_event``, before the busy-state cleanup. B's output
// vanished from the live UI and its sidebar dot stayed stuck busy.
//
// Keying by cwd removes the failure mode rather than patching it:
// there is no global to go stale, so a focus change needs no
// rebinding and every reader names the session it means. Mirrors
// ``pendingByCwd`` here and ``dict[cwd -> SessionRunner]`` on the
// Python side.
const liveTurnByCwd = new Map();

function getLiveTurn(cwd) {
  return cwd ? liveTurnByCwd.get(cwd) || null : null;
}

function setLiveTurn(cwd, turn) {
  if (!cwd) return;
  liveTurnByCwd.set(cwd, turn);
}

function clearLiveTurn(cwd) {
  if (cwd) liveTurnByCwd.delete(cwd);
}

function markVisibleReply(cwd) {
  /* Record that this session's live turn produced something the
   * researcher can see, so the disposable-turn sweep doesn't treat
   * it as a dead end. Keyed by the EMITTING session: a background
   * session's reply must not flip the focused session's flag. */
  const turn = getLiveTurn(cwd || currentCwd);
  if (turn) turn.hasVisibleReply = true;
}

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

async function replayHistory(expectedCwd) {
  /* ``expectedCwd`` pins which session this replay belongs to. After
   * the ``get_chat_history`` await resolves, we compare against the
   * live ``currentCwd`` and abandon if it changed — that means the
   * researcher switched again while our fetch was in flight, and a
   * fresher ``replayHistory`` call is already (or about to be)
   * painting the new session's transcript. Without this guard, a
   * late-arriving response wipes ``messagesEl`` and paints the
   * previous session's history into the now-focused session's view
   * (the response carries whichever cwd was active at call time;
   * pywebview RPC ordering doesn't preserve issue order).
   *
   * Callers that don't pass an explicit cwd default to the live
   * ``currentCwd`` — same effect as the pre-guard behaviour, plus
   * the guard is a no-op (expected matches live).
   */
  if (expectedCwd === undefined) expectedCwd = currentCwd;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.get_chat_history !== 'function') return;
  try {
    const res = await window.pywebview.api.get_chat_history();
    if (expectedCwd !== currentCwd) return;
    if (!res || !res.ok) return;
    const events = res.events || [];
    // Always wipe the existing transcript before painting from
    // the persisted log — even when the log is empty. The empty
    // case fires after a rewind that dropped the very first
    // message: the backend has truncated chat_history.jsonl, but
    // an early ``return`` here would leave the old DOM in place,
    // so ``runEditedMessage``'s subsequent ``appendUser`` would
    // stack the revised bubble on top of stale messages from the
    // dropped branch. Same reasoning for ``setWelcomeOnlyMode`` —
    // the welcome "system" line should also be cleared so the
    // post-rewind state starts empty, and the next live turn
    // toggles welcome mode off naturally.
    messagesEl.innerHTML = '';
    setWelcomeOnlyMode(false);
    if (events.length === 0) return;
    try {
      events.forEach((evt) => replayEvent(evt));
    } finally {
      if (replayTailTurn && !replayTailTurn.hasVisibleReply) {
        queueDisposableTurn(replayTailTurn.nodes);
      }
      replayTailTurn = null;
    }
    scrollToBottom();
    // Restore the context chip from the LAST persisted ``turn_done``,
    // if there is one. Each ``turn_done`` carries the provider's
    // authoritative token counts (``post_turn_tokens`` since the
    // canonical-contract commit; sum of the granular fields for
    // older sessions persisted before that field existed). This is
    // honest data, not a chars/4 estimate. Without it, a session
    // with valid prior measurements would show no context pressure
    // across reload / session-switch / model-swap until another turn
    // completed — which can be a long wait on idle resumes.
    let lastTurnDone = null;
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].type === 'turn_done') {
        lastTurnDone = events[i];
        break;
      }
    }
    // Trigger a fresh pre-flight count after replay. Don't seed the
    // chip from ``lastTurnDone`` usage fields — that mixes
    // post-billing reality into a chip that's supposed to predict
    // the NEXT request's headroom, which was the source of the
    // fluctuation this refactor cleans up.
    triggerContextRecount('replay');
    // Re-anchor the loading indicator if this session is still in
    // flight. ``showChat → syncComposerToFocus`` already added it
    // for a busy session, but the ``messagesEl.innerHTML = ''`` we
    // just ran wiped that node. Without this, a refresh / focus
    // switch into a session mid-turn loses the cat + label until
    // the next event arrives. Check ``busySessions`` (not
    // ``turnInFlight``) because turn_done may have landed mid-
    // replay and flipped the focused-session bit off.
    if (currentCwd && busySessions.has(currentCwd)) {
      showLoadingIndicator();
    }
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
      // ``attachments`` is a list of script filenames (the backend
      // persists the names that travelled with this message).
      const att = Array.isArray(evt.attachments) ? evt.attachments : [];
      // ``images`` is the persisted ``[{data, mime}]`` list. Convert
      // to the data-URL shape ``appendUser`` expects so the bubble
      // shows the actual thumbnails on reload, not a phantom "you
      // sent an image" with no preview. Older histories carry only
      // ``image_count`` (no bytes) — fall through to the no-images
      // path; better a bare text bubble than a broken thumb.
      const imgs = Array.isArray(evt.images)
        ? evt.images
            .filter((i) => i && typeof i.data === 'string')
            .map((i) => ({
              url: `data:${i.mime || 'image/png'};base64,${i.data}`,
              mime: i.mime || 'image/png',
            }))
        : [];
      const userEl = appendUser(evt.text || '', att, imgs);
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
   * ``/Users/you/.nora-sessions/20260422T160059Z_f13630f4`` —
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

// Auto-dismiss handle shared by setLandingBusy and setLandingError.
// Lives at module scope so a busy / new-error transition can cancel
// the prior error's pending clear and avoid wiping a still-relevant
// message mid-action.
let landingErrorTimer = null;

function setLandingBusy(busy, msg) {
  if (landingErrorTimer) {
    clearTimeout(landingErrorTimer);
    landingErrorTimer = null;
  }
  chooseFilesBtn.disabled = busy;
  chooseFolderBtn.disabled = busy;
  landingStatus.classList.remove('error');
  landingStatus.textContent = msg || '';
}

function setLandingError(msg, { autoDismissMs } = {}) {
  if (landingErrorTimer) {
    clearTimeout(landingErrorTimer);
    landingErrorTimer = null;
  }
  chooseFilesBtn.disabled = false;
  chooseFolderBtn.disabled = false;
  landingStatus.classList.add('error');
  landingStatus.textContent = msg;
  if (autoDismissMs) {
    landingErrorTimer = setTimeout(() => {
      landingStatus.classList.remove('error');
      landingStatus.textContent = '';
      landingErrorTimer = null;
    }, autoDismissMs);
  }
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
  // Size gate: reject before FileReader runs. Without this, a
  // researcher who drops a multi-GB .dta sees the app freeze for
  // tens of seconds (and possibly OOM) before the Python side
  // returns "too large." The native picker is one click away and
  // copies straight from disk.
  const oversize = accepted.find((f) => f.size > MAX_DRAG_DROP_BYTES);
  if (oversize) {
    setLandingError(
      formatDragDropOversizeReason(oversize, 'Use Choose files instead.'),
      { autoDismissMs: 4000 },
    );
    return;
  }
  // Aggregate cap: each file passes the per-file cap (above), but
  // the loop below accumulates every file's base64 string in
  // memory before calling upload_files. Two 800 MB files would
  // each pass the per-file cap yet hold ~2.1 GB of base64 in the JS
  // heap concurrently, freezing or crashing the page. Bound the
  // total too. Same threshold as per-file so the user sees a
  // consistent rule: "the drag-drop path can move up to 1 GB at
  // a time, regardless of how many files."
  const aggregateBytes = accepted.reduce((s, f) => s + f.size, 0);
  if (aggregateBytes > MAX_DRAG_DROP_BYTES) {
    setLandingError(
      'Total drop too large. Use Choose files instead.',
      { autoDismissMs: 4000 },
    );
    return;
  }
  try {
    // Read serially with a progress message so large drops don't
    // look frozen. readAsDataURL loads the whole file into memory —
    // size gated above, so the worst case here is one ~1 GB read.
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

// Drag-drop / paste cap on data and script files. The Python backend
// enforces the same cap (``_DRAG_DROP_MAX_BYTES`` in ui.py); both
// sides must move together. The FileReader → base64 → bridge → decode
// chain peaks at roughly 3–4× the file size in memory, so 1 GB peaks
// around 3–4 GB across the JS heap and the bridge transfer. That's
// fine on a 16 GB Mac and tight (but workable) on 8 GB during the
// transfer window — researchers on entry-tier hardware with multi-GB
// datasets should prefer the native file picker (Choose Files… /
// + button), which uses ``shutil.copy2`` and avoids the in-memory
// round-trip entirely. Above this cap, drag-drop is rejected with a
// clear hint pointing at that path. Picking the same cap on both
// sides means rejecting BEFORE allocating multiple GB — rejecting
// after the bridge transfer defeats the whole point of a cap.
const MAX_DRAG_DROP_BYTES = 1024 * 1024 * 1024;

function formatDragDropOversizeReason(file, hint) {
  return `${file.name} is too large for drag and drop. ${hint}`;
}

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
      chip.title = name + '. Saved in this session and sent with your next message.';
    } else {
      chip.title = name + '. Copied into the session.';
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
        // The next request just shrunk — let the chip reflect it
        // immediately rather than waiting for the next turn.
        triggerContextRecount('attachment-remove');
      }
    });
    chip.appendChild(rm);
    attachmentsEl.appendChild(chip);
  });
}

async function stageImageFile(file) {
  if (!ALLOWED_IMAGE_MIMES.has(file.type)) {
    appendError('Only PNG, JPEG, WebP, and GIF images are supported.');
    return false;
  }
  if (file.size > MAX_IMAGE_BYTES) {
    appendError('Image too large. Max 5 MB per file.');
    return false;
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
  return true;
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
  // Size gate (see MAX_DRAG_DROP_BYTES at the top of this section).
  // The composer drop / paste path also goes through FileReader →
  // base64 → bridge → decode, so a multi-GB drop here would freeze
  // the chat the same way it freezes the landing page. The "+"
  // button next to the composer uses the native picker and has no
  // size limit.
  if (file.size > MAX_DRAG_DROP_BYTES) {
    appendError(
      formatDragDropOversizeReason(file, 'Use the + button instead.'),
    );
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
    const uploadCwd = currentCwd;
    const res = await window.pywebview.api.add_files_from_blobs([
      { name: file.name, content: data, mime: file.type || '' },
    ]);
    // The backend puts the bytes in the right session. But the
    // receipts below paint the focused one, so skip them if focus
    // moved — otherwise another session's file shows up here.
    const stillFocused = (res && res.cwd)
      ? res.cwd === currentCwd
      : uploadCwd === currentCwd;
    if (!res || !res.ok) {
      appendError(friendlyAddFilesError(res && res.reason ? res.reason : 'unknown'));
      return;
    }
    if (!stillFocused) {
      // Landed in a session the researcher left. Say where it went.
      const names = (res.added || []).join(', ');
      if (names) {
        toast('Added ' + names + ' to the previous session.', 'info');
      }
      if (typeof loadSessions === 'function') loadSessions();
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
    // Topbar pill shows the cwd path now (not the auto-derived
    // session title), so add_files no longer needs to repaint it —
    // the path doesn't change on upload. The new dataset name
    // surfaces in the sidebar row + Permission chip already.
    refreshFilesChip();
    if (typeof loadSessions === 'function') loadSessions();
    // Script attachments inflate the next request — recount so the
    // chip reflects them. The bridge reads
    // ``runner.pending_script_attachments`` directly.
    triggerContextRecount('attachment-add');
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
        '. Only images and data/script files (.csv, .tsv, .dta, .rds, .parquet, .jsonl, .do, .r, .py, .ipynb, .log, .smcl, .gph, .rmd) can be dropped here.'
      );
    }
    for (const file of usable) {
      if (ALLOWED_IMAGE_MIMES.has(file.type)) {
        // Stage for one-turn vision AND persist to the session cwd
        // so the model can @-mention or read_attached_file the
        // image on later turns. Earlier behavior was vision-only,
        // which made dropped images one-shot while native "+ Add
        // Files" persisted them — confusing inconsistency.
        //
        // Critical: only persist via stageDataFile WHEN the image
        // passed the vision cap. Earlier code unconditionally ran
        // both, so a 100 MB screenshot rejected by the 5 MB vision
        // cap STILL got FileReader-base64'd and bridge-sent under
        // the 1 GB drag-drop cap, freezing the UI on the very
        // payload the image cap was meant to refuse. Treat the
        // image cap as the floor for both paths.
        const accepted = await stageImageFile(file);
        if (accepted) {
          await stageDataFile(file);
        }
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
        // drop handler. The image-cap-rejection short-circuit also
        // applies here: an oversize pasted screenshot must not slip
        // through the data path's larger cap. See the drop handler
        // above for the full rationale.
        const accepted = await stageImageFile(f);
        if (accepted) {
          await stageDataFile(f);
        }
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
    // Keep the selected row in view when arrow-key navigation
    // moves past the visible window. ``.mention-popup`` is the
    // scroll container (max-height 280px, overflow-y: auto), and
    // without this the highlighted row slides off the top/bottom
    // as the user holds ↓ — the visual selection just disappears.
    // ``block: 'nearest'`` scrolls only when the row is actually
    // off-screen, so rows that are already in view don't bounce.
    const selectedRow = list.querySelector('.mention-row.selected');
    if (selectedRow && typeof selectedRow.scrollIntoView === 'function') {
      selectedRow.scrollIntoView({ block: 'nearest' });
    }
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
  // Pass the row's path through to the bridge so basename
  // collisions (helper plots in different run dirs commonly
  // produce ``coefficients.png`` or ``marginal_effects.png``)
  // resolve to the exact row the researcher clicked, not
  // whichever copy ``iterdir`` returns first.
  await stageMentionedFile(item.name, item.path);
  input.focus();
}

async function stageMentionedFile(name, path) {
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.attach_session_file !== 'function') return;
  try {
    const res = await window.pywebview.api.attach_session_file(name, path);
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

// ---- hard reload (Cmd/Ctrl+Shift+R) --------------------------------------
//
// In-app reload (Cmd+R) re-fetches the SAME ``index.bust-<id>.html``
// URL the bridge wrote at startup, so ``style.css?v=<old-id>`` and
// ``app.js?v=<old-id>`` hit WKWebView's persistent disk cache —
// edits made while nora is running don't show up. Hard reload calls
// the bridge's ``hard_reload``, which recomputes the build-id from
// the current asset mtimes, writes a fresh ``.bust-*.html``, and
// navigates the window to its file:// URL. New URL → cache miss →
// fresh fetch → CSS/JS edits visible without quitting nora.
//
// Bound to Cmd+Shift+R (macOS convention) AND Ctrl+Shift+R (so a
// keyboard with no Cmd, or future non-mac builds, keep working).
// Both browsers and editors use this combo for "force-reload, ignore
// cache," which is exactly the semantics here.
document.addEventListener('keydown', (e) => {
  if (!e.shiftKey) return;
  if (!(e.metaKey || e.ctrlKey)) return;
  if (e.key !== 'R' && e.key !== 'r') return;
  e.preventDefault();
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.hard_reload !== 'function') return;
  // Fire-and-forget. The bridge calls load_url which navigates the
  // window away from the current page, so any logging here would be
  // racing against the navigation. The new page picks up logging on
  // its own once it loads.
  window.pywebview.api.hard_reload();
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
  // A queued message can fire AFTER the researcher has switched
  // sessions. The turn is registered under ITS OWN cwd, so a
  // background flush can't disturb the focused session's live turn —
  // Stop, hasVisibleReply, and the disposable-turn cleanup each look
  // up by cwd and get their own. ``isFocused`` below still gates the
  // error bubbles, which are a visible-transcript surface.
  const isFocused = (cwd === currentCwd);
  const localTurn = { id: null, nodes: [item.userEl], hasVisibleReply: false };
  setLiveTurn(cwd, localTurn);
  // Use the explicit-target send variants. The plain ``send_message``
  // routes to the bridge's ``self.cwd`` (focused session); a queue
  // can flush AFTER the user has switched sessions, so falling back
  // to the focused cwd would persist / execute the queued message
  // against the WRONG session. The ``_to_session`` variants route to
  // the runner whose cwd matches ``cwd``, regardless of focus.
  //
  // ``item.attachmentToken`` is the per-queued-message attachment
  // snapshot the bridge took at queue time. The token-bearing send
  // variants restore that exact snapshot into runner.pending_* just
  // before sending, so two queued messages with different staged
  // scripts each fire with their OWN attachments rather than racing
  // for whatever happens to be in the global pending lists.
  const api = window.pywebview.api;
  const supportsTokenedSend = (
    typeof api.send_message_with_token_to_session === 'function'
    && typeof api.send_message_with_images_and_token_to_session === 'function'
  );
  const supportsTargeted = (
    typeof api.send_message_to_session === 'function'
    && typeof api.send_message_with_images_to_session === 'function'
  );
  try {
    let turnId = null;
    const token = item.attachmentToken || '';
    if (item.images.length > 0) {
      const payload = item.images.map((img) => ({ data: img.data, mime: img.mime }));
      if (supportsTokenedSend && token) {
        turnId = await api.send_message_with_images_and_token_to_session(
          cwd, item.text, payload, token,
        );
      } else if (supportsTargeted) {
        turnId = await api.send_message_with_images_to_session(cwd, item.text, payload);
      } else if (typeof api.send_message_with_images === 'function') {
        // Older bridge: fall back to focused-cwd send. Cross-session
        // mix-up risk remains until the new APIs ship; the explicit
        // path above is the durable fix.
        turnId = await api.send_message_with_images(item.text, payload);
      } else {
        // The error bubble is a focused-transcript surface; only
        // surface it (and queue disposable nodes against the focused
        // global) when this flush is for the focused session.
        if (isFocused) {
          const errEl = appendError('Restart Nora to send images.');
          localTurn.nodes.push(errEl);
          queueDisposableTurn(localTurn.nodes);
        }
        clearLiveTurn(cwd);
        setSending(false, cwd);
        return;
      }
    } else if (supportsTokenedSend && token) {
      turnId = await api.send_message_with_token_to_session(cwd, item.text, token);
    } else if (supportsTargeted) {
      turnId = await api.send_message_to_session(cwd, item.text);
    } else {
      turnId = await api.send_message(item.text);
    }
    localTurn.id = turnId;
  } catch (err) {
    // Same focused-only routing as the no-image-support branch above.
    if (isFocused) {
      const errEl = appendError('send failed: ' + err);
      localTurn.nodes.push(errEl);
      queueDisposableTurn(localTurn.nodes);
    }
    clearLiveTurn(cwd);
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
  const api = window.pywebview && window.pywebview.api;
  for (const item of q) {
    item.userEl.classList.remove('queued');
    item.userEl.classList.add('not-sent');
    item.userEl.title = 'Stopped before this message could send.';
    // Drop each cancelled item's frozen attachment snapshot from
    // the runner — without this, a Stop+resend cycle would leak
    // the snapshots and any attachments restored later by a
    // misrouted token would carry stale state.
    if (
      item.attachmentToken
      && api
      && typeof api.discard_pending_attachments_token === 'function'
    ) {
      try {
        api.discard_pending_attachments_token(cwd, item.attachmentToken);
      } catch (err) {
        console.warn('discard_pending_attachments_token failed', err);
      }
    }
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
    userEl.title = 'Queued. Will send when the current turn finishes.';
    // Freeze the runner's current pending_* lists into a per-message
    // snapshot. The bridge clears the runner's pending state, so any
    // attachments the user stages NEXT (for a later queued message)
    // land in a fresh slot. When this queued item fires below, the
    // backend will restore THIS snapshot into pending_* — closing
    // the race where the second queued send used to consume the
    // first message's script chip.
    let attachmentToken = '';
    const api = window.pywebview && window.pywebview.api;
    if (api && typeof api.freeze_pending_attachments === 'function') {
      try {
        const t = await api.freeze_pending_attachments(currentCwd);
        if (typeof t === 'string') attachmentToken = t;
      } catch (err) {
        // If freezing fails for any reason, fall through to the
        // older fire-without-token path. The race is back, but the
        // message at least sends — silent failure here would be
        // worse for the user.
        console.warn('freeze_pending_attachments failed', err);
      }
    }
    pendingFor(currentCwd).push({
      text,
      images: images.map((img) => ({ data: img.data, mime: img.mime })),
      attachments: messageAttachments,
      attachmentToken,
      userEl,
    });
    return;
  }

  // No "clear cancelled state" step: each turn has its own id, and
  // the new turn's id won't be in ``cancelledTurnIds``. Late events
  // from the previously-cancelled turn keep getting dropped because
  // they carry the OLD id; new events flow because they carry the
  // NEW id. That's the whole point of the turn-identity rewrite.
  //
  // Capture the cwd this send belongs to: ``currentCwd`` can change
  // while the send await is in flight (the researcher switches
  // sessions), so every write below names the session that actually
  // sent rather than whichever happens to be focused when the
  // promise settles.
  const sendCwd = currentCwd;
  const localTurn = { id: null, nodes: [userEl], hasVisibleReply: false };
  setLiveTurn(sendCwd, localTurn);
  setSending(true);
  try {
    // If images are attached, use the richer send method. The
    // simpler string send stays as the fast path for text-only.
    let turnId = null;
    if (images.length > 0 && typeof window.pywebview.api.send_message_with_images === 'function') {
      const payload = images.map((img) => ({ data: img.data, mime: img.mime }));
      turnId = await window.pywebview.api.send_message_with_images(text, payload);
    } else if (images.length > 0) {
      const errEl = appendError('Restart Nora to send images.');
      if (!localTurn.hasVisibleReply) {
        localTurn.nodes.push(errEl);
        queueDisposableTurn(localTurn.nodes);
      }
      clearLiveTurn(sendCwd);
      setSending(false);
      return;
    } else {
      turnId = await window.pywebview.api.send_message(text);
    }
    // Capture the turn id the bridge assigned. Stop reads this to
    // mark the turn cancelled. If the await resolved with no id
    // (unexpected: the bridge returns null only on early failure
    // paths that already dispatched a turn_error), leave id null
    // and let the turn_error event clean up.
    localTurn.id = turnId;
    // Don't clear setSending here — the await resolves as soon as
    // the turn is QUEUED on the Python side, not when it finishes.
    // The turn_done / turn_error / auth_failure event handler
    // below is what flips the button back on.
  } catch (err) {
    const errEl = appendError('send failed: ' + err);
    if (!localTurn.hasVisibleReply) {
      localTurn.nodes.push(errEl);
      queueDisposableTurn(localTurn.nodes);
    }
    clearLiveTurn(sendCwd);
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
  'State your hypothesis',
  'What are we estimating today?',
  'Ask me something causal',
  'One regression at a time',
  'Ask, regress, repeat',
  'What shall we regress today?',
  'Show me a variable',
  'Feed me a question',
  'Ask away, researcher',
  'Ready when you are',
  'What’s the question?',
  'What a beautiful day',
  'AMA',
  'Where should we start?',
  'What are you working on?',
  'Tell me what you need',
  'Let’s take a look',
  'What’s on your mind?',
  'What’s your estimand?',
  'Name your outcome',
  'Choose your treatment',
  'Power calculations on request',
  'The model is caffeinated',
  'Cluster me softly',
  'Propensity scores and good intentions',
  'The p-value trembles',
  'Frequentist by day, Bayesian by night',
  'I do not fear missingness',
  'Endogeneity hotline',
  'Heteroskedasticity-robust greetings',
  'Placebo tests for the soul',
  'DiD or it didn’t happen',
  'May your joins be many-to-one',
  'Bring me your messy joins',
  'Long or wide, I do not judge',
  'Reshape me like one of your French panels',
  'Fixed effects, flexible morals',
  'Panel data, biscuits, tea',
  'I read code so you don’t have to',
  'Surrender the .dta',
  'Securely curious',
  'I come in privacy',
  'Nothing raw leaves the room',
  'No raw access, all the insight',
  'Your data stays home tonight',
  'Ask without peeking',
  'A question walks into a dataset',
  'The answer is in there somewhere',
  'The data is restless tonight',
  'Tell me where it hurts, in the data',
  'Poke the dataset',
  'Reviewer 2 is asleep, talk to me',
  'Postcard from the ivory tower',
  'The replication crisis sends regards',
  'Coffee or coefficient?',
  'Pun buffer: loaded',
  'Identification, please',
  'What a fine day for a join',
  'Nam nam nam data…',
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
    // No cancelled-state clearing here. Each turn has its own id
    // and the new turn's events naturally pass the
    // ``cancelledTurnIds`` filter without needing a per-cwd reset
    // (that reset was the prior leak: it allowed late events from
    // the previous, cancelled turn to render after a fresh send
    // started). The cancelled set is only mutated by Stop, never
    // by Send.
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
  // Data-massage flavor — same gerund shape, vaguely statistical
  // verbs that sound like they describe what's happening to the
  // numbers without committing to any specific tool call. Bare
  // verb when the verb stands alone; object retained only where
  // the noun IS the joke (e.g. "reticulating splines").
  'massaging',
  'crunching',
  'wrangling',
  'kneading',
  'sifting',
  'whisking',
  'polishing',
  'coaxing',
  'shuffling',
  'tightening',
  'auditing',
  'interrogating',
  'sweeping',
  'tuning',
  'weighing',
  'herding outliers',
  'minding the gaps',
  'reticulating splines',
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
  appendFollowing(el);
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
    // Mark this turn's events as suppressed BEFORE awaiting the
    // bridge cancel. Two ids go into ``cancelledTurnIds``, belt and
    // suspenders:
    //   1. The FOCUSED session's live turn id, captured when its
    //      send Promise resolved. Available unless Stop fires in the
    //      tiny window between Send and the await returning. Looked
    //      up by cwd: another session may have sent more recently,
    //      and blacklisting its id here would silence a session that
    //      is still legitimately running.
    //   2. The id the bridge returns from ``interrupt_turn``. The
    //      bridge knows which turn it just cancelled and surfaces
    //      the id explicitly. Covers the race above.
    // Either path alone would usually be enough; together they
    // guarantee the JS-side filter has the cancelled id no matter
    // when Stop fires relative to the send Promise.
    const stoppingTurn = getLiveTurn(currentCwd);
    if (stoppingTurn) markTurnCancelled(stoppingTurn.id);
    // The terminal turn_error that the runner emits is now also
    // dropped (see ``nora_event`` for why), so handle the live-turn
    // cleanup the terminal handler used to do. If the cancelled
    // turn produced no visible reply (the model hadn't written
    // anything when Stop fired), queue the user-bubble nodes for
    // the staleness sweep so they don't accumulate.
    if (stoppingTurn && !stoppingTurn.hasVisibleReply) {
      queueDisposableTurn(stoppingTurn.nodes);
    }
    clearLiveTurn(currentCwd);
    // Stop = "stop everything for this session": cancel the
    // running turn AND drain any queued follow-ups. Cancelled
    // queued messages get marked ``.not-sent`` so the researcher
    // can see what they typed but didn't ship; silently removing
    // their text would be hostile UX.
    drainPendingFor(currentCwd);
    if (currentCwd) {
      triggerContextRecount('stop');
    }
    // Visible acknowledgement: the cancellation cascades through
    // the SDK + the subprocess kill, which can take a beat. Without
    // this toast, a researcher who pressed Stop and immediately
    // resent the same prompt would see the new turn queue behind
    // the cancellation cleanup and assume "Stop did nothing".
    toast('Stopping…', 'info');
    try {
      const res = await window.pywebview.api.interrupt_turn();
      // Belt-and-suspenders: if the bridge tells us which turn id
      // it cancelled, add that to the drop set too. Covers the
      // case where the focused session had no live turn at click time
      // (e.g., Stop fired between fireQueuedMessage clearing the
      // turn and the next message starting).
      if (res && res.turn_id) markTurnCancelled(res.turn_id);
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

  // Drop late events from any turn the researcher cancelled.
  //
  // Why drop ALL events from the cancelled turn (not "all except the
  // terminal"): cancellation is async at multiple levels — the
  // asyncio task gets a cancel signal, the provider stream still
  // has buffered tokens, and pywebview's evaluate_js queue has a
  // tail of events from before the cancel. Earlier behavior keyed
  // suppression on the cwd, then cleared it the moment a new
  // message started — late events from the previous turn then
  // slipped through and rendered after the new turn began. The
  // turn-id key fixes that: each turn has a unique id, the
  // cancelled set is never cleared by Send, and a new turn's
  // events naturally pass because they carry a fresh id.
  //
  // The backend dispatcher (``_dispatch_event`` in ui.py) applies
  // the same drop authoritatively — this filter is best-effort
  // defense in depth. If both layers ever disagree, the backend
  // wins (events that don't reach JS were never persisted).
  if (evt.turn_id && cancelledTurnIds.has(evt.turn_id)) {
    return;
  }

  switch (evt.type) {
    case 'ready':
      showChat(evt);
      break;
    case 'assistant_text':
      if (!isFocused) return;
      markVisibleReply(evtCwd);
      appendAssistant(evt.text);
      break;
    case 'assistant_thinking':
      if (!isFocused) return;
      markVisibleReply(evtCwd);
      appendThinking(evt.text);
      break;
    case 'tool_call': {
      if (!isFocused) return;
      const card = appendToolCall(evt);
      if (card) markVisibleReply(evtCwd);
      break;
    }
    case 'tool_result': {
      if (!isFocused) return;
      const card = appendToolResult(evt);
      if (card) markVisibleReply(evtCwd);
      // Refresh the Files panel only when the tool that just finished
      // could plausibly have written to disk. ``run_dir`` is populated
      // by the provider layer iff this result came from
      // ``submit_script`` / ``submit_script_file`` — the only tools
      // that stage a script, log, or plot. Plumbing tools
      // (``get_schema``, ``expand_result``, ``list_results``,
      // ``request_data``) leave it ``null`` and don't need a refresh.
      //
      // The cost matters: ``list_session_files`` walks every run
      // dir under ``.nora/runs``, stats each ``script.{do,R,py}``,
      // recurses into every ``_nora_plots/`` subdir, and base64-
      // encodes thumbnails up to 3 MB each. In a long session that
      // adds up fast, and firing it on every plumbing tool turned
      // the chat loop into a steady drip of heavy IPC.
      if (evt.run_dir) refreshFilesChip();
      break;
    }
    case 'turn_done':
      // Terminal event: clear busy state for THIS session
      // (whether focused or background) and, if focused, refresh
      // the composer + context chip.
      //
      // The chip tracks "context occupied AFTER this turn" — i.e.,
      // the prompt this turn loaded PLUS the response that just
      // landed. Each provider computes ``post_turn_tokens`` from its
      // own usage fields (Anthropic sums input + cache_read +
      // cache_creation + output; OpenAI sums input + output because
      // its input_tokens already covers the cached prefix), so the
      // chip just renders that number. Older sessions persisted
      // before this field existed fall back to the legacy sum below.
      if (isFocused) {
        // Refresh the chip from the pre-flight counter — single
        // source of truth. The provider's ``post_turn_tokens`` is
        // post-billing reality, useful for diagnostics but not the
        // right number for "what's the next request going to weigh"
        // (which is what the chip predicts). Logged below for any
        // debug consumer that still wants the post-hoc value.
        if (typeof evt.post_turn_tokens === 'number') {
          // Diagnostic only; not on the chip.
        }
        triggerContextRecount('turn_done');
        const doneTurn = getLiveTurn(evtCwd || currentCwd);
        if (doneTurn && !doneTurn.hasVisibleReply) {
          queueDisposableTurn(doneTurn.nodes);
        }
      }
      // Retire the entry whether or not this session is focused, so a
      // background session's finished turn can't linger in the map
      // and be mistaken for live work later.
      clearLiveTurn(evtCwd || currentCwd);
      // ``flushPendingFor`` returns true iff a queued message just
      // fired. In that case ``setSending(true, evtCwd)`` was
      // re-asserted inside, so we leave the composer in the busy
      // state and DON'T flip back to Send.
      if (!flushPendingFor(evtCwd)) {
        setSending(false, evtCwd);
      }
      // Deliberately no scroll here. The turn ending is not a
      // reason to move the researcher's viewport: they may be
      // mid-read somewhere above. hideLoadingIndicator() removes
      // the cat and shifts content up by ~80px, so refresh the
      // "scroll to latest" button against the new geometry.
      updateScrollToBottomVisibility();
      // Refresh the sidebar so the just-active session bubbles to
      // the top — list_sessions sorts by chat_history.jsonl mtime,
      // which the persist path bumped during this turn.
      if (typeof loadSessions === 'function') loadSessions();
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
        const failedTurn = getLiveTurn(evtCwd || currentCwd);
        if (failedTurn && !failedTurn.hasVisibleReply) {
          failedTurn.nodes.push(errEl);
          queueDisposableTurn(failedTurn.nodes);
        }
      }
      clearLiveTurn(evtCwd || currentCwd);
      drainPendingFor(evtCwd);
      setSending(false, evtCwd);
      if (evtCwd === currentCwd) triggerContextRecount('turn_settled');
      break;
    case 'turn_error':
      if (isFocused) {
        const errEl = appendError(evt.message || 'unknown error');
        const failedTurn = getLiveTurn(evtCwd || currentCwd);
        if (failedTurn && !failedTurn.hasVisibleReply) {
          failedTurn.nodes.push(errEl);
          queueDisposableTurn(failedTurn.nodes);
        }
      }
      clearLiveTurn(evtCwd || currentCwd);
      // For ordinary turn errors (e.g., model returned a tool-use
      // error), drain the queue: the researcher's follow-ups were
      // probably reasoning-conditioned on the previous turn
      // succeeding, so firing them blindly is worse than asking
      // them to retry.
      drainPendingFor(evtCwd);
      setSending(false, evtCwd);
      if (evtCwd === currentCwd) triggerContextRecount('turn_settled');
      break;
    case 'policy_updated':
      updatePolicyChip(evt.policy);
      break;
    case 'install_confirmation_request':
      // Hard consent gate for the install_packages tool. The Python
      // handler is blocked awaiting a response keyed by ``evt.token``;
      // a missing emitter would have made it deny immediately, so the
      // request only arrives here when a real install is pending.
      // Show the modal regardless of focused vs background cwd: the
      // install affects the researcher's machine globally, so the
      // confirmation belongs in front of them right now.
      showInstallConfirmationModal(evt);
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
  return append('assistant', text || '', /*markdown=*/ true);
}

function appendThinking(text) {
  /* Claude's reasoning trace. Rendered as a collapsible card so the
   * researcher can see WHAT Claude was thinking without the trace
   * dominating the transcript. Starts collapsed; clicking the
   * header expands. Mirrors the submit_script card shape so the
   * interaction model is consistent. */
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
  appendFollowing(card);
  return card;
}

function appendSystem(text) {
  return append('system', text);
}

function appendError(text) {
  return append('error', text);
}

function append(kind, text, markdown, attachments, images) {
  setWelcomeOnlyMode(false);
  const wrapper = document.createElement('div');
  wrapper.className = 'message ' + kind;
  // Stash the original (pre-render) text on the wrapper so the
  // per-bubble copy button can hand the markdown source — not the
  // rendered HTML — to the clipboard. Researchers paste these into
  // other chats / editors and want the literal text they sent or
  // received, not a transformed version with HTML entities decoded
  // / list bullets converted to glyphs / etc. Stored as a property
  // (not a dataset attribute) to avoid a large stringify on every
  // long assistant turn.
  wrapper.__noraRawText = text || '';
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
  // Copy button — only on user and assistant bubbles. System / error
  // / thinking messages don't get one: system + error are rare status
  // lines (clipboard isn't useful), and thinking traces are already
  // in a collapsible card with their own controls. The button hides
  // by default and reveals on bubble hover via the ``.message-actions``
  // / ``.message:hover .message-actions`` rules in style.css; on
  // touch / no-hover devices the action row is always visible (a
  // matching ``@media (hover: none)`` rule keeps it on those clients).
  if (kind === 'user' || kind === 'assistant') {
    const actions = document.createElement('div');
    actions.className = 'message-actions';
    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'message-action-btn message-copy-btn';
    copyBtn.title = 'Copy message to clipboard';
    copyBtn.setAttribute('aria-label', 'Copy message');
    // Inline SVG so the icon doesn't depend on a font load. Same
    // copy glyph as the Files-panel action; see ``iconSvg``.
    copyBtn.innerHTML = iconSvg('copy');
    copyBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      copyMessageBubble(wrapper, copyBtn);
      // Drop focus so the actions row hides when the cursor leaves
      // the message — without this, the button keeps focus and any
      // ``:focus-within``-style rule would pin the row visible after
      // a mouse click.
      copyBtn.blur();
    });
    actions.appendChild(copyBtn);
    // Edit button — user bubbles only. Provenance boundary: editing
    // an assistant bubble would let the researcher put words in
    // the model's mouth, then have the chat continue from there as
    // if the model had said them. That's a shape we deliberately
    // refuse. User-message edit is the supported "revisit and try
    // a different question" affordance; the rewind path truncates
    // the chat at THIS user message, hides results from the dropped
    // branch in the store, and re-fires the bubble's new text as a
    // fresh turn with a warm-start replay over the truncated log.
    if (kind === 'user') {
      const editBtn = document.createElement('button');
      editBtn.type = 'button';
      editBtn.className = 'message-action-btn message-edit-btn';
      editBtn.title = 'Edit and re-run from this message';
      editBtn.setAttribute('aria-label', 'Edit message');
      // Pencil glyph — same stroke style as the copy icon for
      // visual rhythm in the actions row.
      editBtn.innerHTML = iconSvg('edit');
      editBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        enterEditMode(wrapper);
      });
      actions.appendChild(editBtn);
    }
    wrapper.appendChild(actions);
  }
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
      chip.title = name + '. Sent with this message.';
      row.appendChild(chip);
    });
    wrapper.appendChild(row);
  }
  const wasFollowing = beginFollowingAppend();
  messagesEl.appendChild(wrapper);
  if (kind === 'user') {
    // The researcher just hit Send — always take them to their own
    // message. This is the one scroll that's unconditional, because
    // it's a direct response to their action. Passing ``true``
    // rather than calling scrollToBottom() directly also refreshes
    // the button, which matters because the instant scroll fires no
    // scroll event to trigger that refresh on its own.
    endFollowingAppend(true);
  } else {
    // Everything that arrives on its own (assistant replies, system
    // and error notices) follows only if the researcher is already
    // parked at the bottom. If they've scrolled up to read something
    // earlier, their position is left alone and the
    // "scroll to latest" button surfaces instead.
    //
    // This replaces an earlier top-anchoring scheme that pinned each
    // incoming reply under the user's preceding question. It read
    // well for a single short exchange, but a turn that emits
    // several assistant messages re-anchored on the SAME originating
    // question every time, so the view kept getting yanked back up
    // the transcript mid-read.
    endFollowingAppend(wasFollowing);
  }
  return wrapper;
}

/* Auto-follow: keep chasing new content only while the researcher is
 * parked at the end of the transcript. If they've scrolled up to read
 * something, incoming replies must not yank the viewport away.
 *
 * Both halves of the measurement are load-bearing and easy to get
 * wrong:
 *
 * 1. "Were they at the bottom" MUST be sampled BEFORE the new node
 *    enters the DOM. Measured after, the just-appended message's own
 *    height counts as distance-from-bottom, so any reply taller than
 *    the threshold reads as "they scrolled away" and following dies
 *    after the first message of every turn.
 * 2. It must be a live geometry read, not a flag cached from scroll
 *    events. Programmatic scrolls don't reliably emit scroll events
 *    in WKWebView, so a flag maintained that way silently goes stale.
 *
 * ``endFollowingAppend`` also refreshes the button, because appending
 * grows scrollHeight without moving scrollTop — no scroll event
 * fires, so the button would otherwise stay hidden while new content
 * piles up below the fold.
 */
function beginFollowingAppend() {
  return isNearBottom();
}

function endFollowingAppend(wasFollowing) {
  if (wasFollowing) scrollToBottom();
  updateScrollToBottomVisibility();
}

function appendFollowing(node) {
  /* appendChild + auto-follow, sampled in the right order. */
  if (!messagesEl || !node) return;
  const wasFollowing = beginFollowingAppend();
  messagesEl.appendChild(node);
  endFollowingAppend(wasFollowing);
}

function appendToolCall(evt) {
  // Only ``submit_script`` and ``submit_script_file`` render cards.
  // ``get_schema``, ``request_data``, ``expand_result``,
  // ``list_results`` etc. happen silently — they're plumbing, not
  // results the researcher reads. Claude summarizes whatever
  // matters from them in the chat text that follows.
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
  appendFollowing(card);
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
  const wasFollowing = beginFollowingAppend();
  renderScriptResultInline(body, evt);
  endFollowingAppend(wasFollowing);
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
  /* Jump (don't animate) to the end of the transcript.
   *
   * ``.messages`` sets ``scroll-behavior: smooth`` in CSS, which
   * makes a bare ``scrollTop = scrollHeight`` start an animation
   * rather than move immediately. That breaks auto-follow: the next
   * message to land reads a mid-animation scrollTop, concludes the
   * researcher has scrolled away, and stops following — so a burst
   * of replies ends up stranded near the top. Forcing ``instant``
   * keeps the position truthful the moment we set it, so
   * ``isNearBottom()`` can be trusted on the very next append.
   *
   * The "scroll to latest" button deliberately keeps ``smooth``: a
   * click IS a navigation gesture, and there is no follow-up read
   * racing against it.
   */
  if (typeof messagesEl.scrollTo === 'function') {
    messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'instant' });
  } else {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

// "Scroll to latest" floating button. Anchored above the composer
// in the markup; visibility tracks whether the transcript is near
// its bottom.
const scrollToBottomBtn = document.getElementById('scroll-to-bottom');
const SCROLL_TO_BOTTOM_THRESHOLD = 100;

function isNearBottom() {
  /* True when the transcript is scrolled to (or very near) its end.
   * Single source of truth for both the "scroll to latest" button's
   * visibility and whether incoming content auto-follows. */
  if (!messagesEl) return true;
  const distanceFromBottom = (
    messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight
  );
  return distanceFromBottom < SCROLL_TO_BOTTOM_THRESHOLD;
}

function updateScrollToBottomVisibility() {
  if (!scrollToBottomBtn || !messagesEl) return;
  // Welcome screen has no transcript to scroll through, so the
  // button would just be visual noise above the composer.
  if (messagesEl.classList.contains('welcome-only')) {
    scrollToBottomBtn.classList.add('hidden');
    return;
  }
  scrollToBottomBtn.classList.toggle('hidden', isNearBottom());
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
    //
    // Note: smooth programmatic scrolls are ignored outright in some
    // non-WebKit engines, which makes this look dead when the UI is
    // driven in a test browser. It works in WKWebView, which is what
    // ships, so no fallback is wired here — a timeout-based hard
    // land would interrupt the real animation on a long transcript.
    if (typeof messagesEl.scrollTo === 'function') {
      messagesEl.scrollTo({
        top: messagesEl.scrollHeight,
        behavior: 'smooth',
      });
    } else {
      scrollToBottom();
    }
    updateScrollToBottomVisibility();
  });
}

function setWelcomeOnlyMode(enabled) {
  if (!messagesEl) return;
  messagesEl.classList.toggle('welcome-only', !!enabled);
  updateScrollToBottomVisibility();
}

// ----- policy chip + popup (next to composer) ----------------------------

const policyChip = document.getElementById('policy-chip');
const policyChipLabel = document.getElementById('policy-chip-label');
const policyPopup = document.getElementById('policy-popup');
let policyPopupBuiltFor = null;  // cached copy so we don't rebuild needlessly

function renderContextChip() {
  /* Paint the chip from ``lastContextCount`` (the last solid count
   * returned by the backend). Stable single-source-of-truth render:
   * no projections, no pending-message arithmetic, no provider
   * ``turn_done`` echoes. The number changes only when a recount
   * response lands — see ``triggerContextRecount`` for the trigger
   * list (turn complete, rewind success, session switch, attachment
   * add/remove).
   *
   * The chip text itself carries NO ``~`` prefix even when
   * ``exact=False`` (chars/3.5 fallback today — replaced by
   * tiktoken / count_tokens once those paths land). The
   * approximate-vs-exact distinction lives in the chip's tooltip
   * instead: an exact count shows just ``"N tokens"``, an
   * approximate count shows ``"N tokens · local estimate, expect
   * a small gap from the provider's billed count"``. This is the
   * only place in the UI that should display a context-size
   * number; ad-hoc estimates elsewhere were the bug we're fixing.
   */
  if (!contextChip) return;
  if (!lastContextCount) {
    contextChip.classList.add('hidden');
    return;
  }
  const { tokens, exact, ceiling } = lastContextCount;
  const fmt = (n) => (n >= 10_000 ? (n / 1000).toFixed(1) + 'k' : n.toString());
  const ceilingLabel = ceiling >= 1_000_000
    ? (ceiling / 1_000_000) + 'M'
    : (ceiling / 1000) + 'k';
  const rawPct = Math.round((tokens / ceiling) * 100);
  // No ``~`` prefix on the chip text. While ``exact`` is False on
  // every render today (the chars/3.5 fallback is the only path
  // wired), a prefix that's always present conveys no information
  // and reads as visual noise. The tooltip still spells out that
  // the value is an approximation. When exact tokenization lands
  // (tiktoken for OpenAI, count_tokens API for Claude), the
  // tooltip wording switches but the chip text stays clean.
  contextChip.textContent =
    `Context ${fmt(tokens)} / ${ceilingLabel} (${rawPct}%)`;
  // Tooltip: precise count + a one-clause honesty caveat when the
  // number is local-approximate. Researchers know how tokens work;
  // they don't need a trigger list. The precise count is the only
  // value-add over the chip text itself ("47.2k" → "47,234").
  contextChip.title = exact
    ? `${tokens.toLocaleString()} tokens`
    : (
        `${tokens.toLocaleString()} tokens · local estimate, ` +
        `expect a small gap from the provider's billed count`
      );
  contextChip.classList.remove('hidden');
  contextChip.classList.remove('warn', 'danger', 'over');
  if (rawPct >= 100) contextChip.classList.add('over');
  else if (rawPct >= 90) contextChip.classList.add('danger');
  else if (rawPct >= 70) contextChip.classList.add('warn');
}

async function triggerContextRecount(reason) {
  /* Refresh the context chip from the backend's pre-flight counter.
   * Call sites: turn complete, rewind success, session open/switch,
   * attachment add/remove. ``reason`` rides through to the console
   * for debugging only — the chip itself shows no "updating..." text.
   *
   * Visual contract: fade the chip immediately so the researcher
   * sees their action acknowledged, even before the backend
   * returns. The fade clears when the matching count response
   * lands. A newer recount (different ``request_id``) supersedes
   * an in-flight one — the older response, when it eventually
   * arrives, fails the id check and is dropped, leaving the chip
   * faded until the newer response lands.
   */
  if (!contextChip) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.count_next_context !== 'function') return;

  contextCountRequestId += 1;
  const id = contextCountRequestId;
  contextChip.classList.add('stale');

  // Pull whatever the composer currently shows. Empty when the chip
  // is fired between turns. The composer element is ``input`` per
  // ``getElementById('compose-input')`` at module top — earlier
  // drafts of this code referenced ``composerEl`` which was never
  // declared, throwing a silent ReferenceError inside this async
  // function and leaving the chip stuck hidden forever.
  const draftText = (input && input.value) || '';
  const nImages = pendingComposerImageCount();
  const nAttachments = pendingComposerScriptCount();

  let res;
  try {
    res = await window.pywebview.api.count_next_context(
      draftText, nImages, nAttachments, id,
    );
  } catch (err) {
    // Network / bridge failure. Leave the chip faded but stop
    // claiming "updating" — the next trigger will retry. The
    // researcher still sees the last solid value, just dimmed.
    console.warn('count_next_context failed', err);
    return;
  }
  if (!res || !res.ok) return;
  if (res.request_id !== contextCountRequestId) {
    // Stale response — newer recount in flight. Drop silently.
    return;
  }
  lastContextCount = {
    tokens: res.tokens,
    exact: !!res.exact,
    ceiling: res.ceiling,
  };
  contextChip.classList.remove('stale');
  renderContextChip();
}

function pendingComposerImageCount() {
  // Pending images attached to the next send (drag-drop / paste).
  // Module-local state, mirrors what ``handleSubmit`` will pack.
  return Array.isArray(stagedImages) ? stagedImages.length : 0;
}

function pendingComposerScriptCount() {
  // Script attachments live on the bridge runner, not in JS state.
  // The bridge's ``count_next_context`` reads
  // ``runner.pending_script_attachments`` directly and overrides
  // whatever JS passes here with the authoritative count + content
  // bytes — so 0 from this side just means "let the backend tell
  // us." Keeping the call site lets a future per-tab JS-only
  // staging list slot in without touching the recount path.
  return 0;
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
      '<strong>Files</strong>. Scripts, graphs, and logs from this '
      + 'session.';
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
   * [title / thumbnail] [delete]. Every actionable kind uses
   * "copy to clipboard" — the verb stays consistent across rows
   * so a researcher can grab any output and paste it into another
   * chat or an external editor without context-switching to the
   * folder. The clipboard payload varies with the file kind:
   *   - graph (image with thumbnail data): copy as image
   *   - graph (no thumbnail data, e.g. .gph): open externally
   *     (no meaningful clipboard representation for a Stata-
   *     binary plot file)
   *   - script: copy file text content
   *   - log:    copy file text content
   * Delete is always on the right.
   *
   * Why copy instead of "send to next message" for scripts:
   * scripts already get attached via the @-mention dropdown and
   * via drag-drop, so the panel button doesn't need to duplicate
   * that path. Copy-to-clipboard solves the OTHER common case —
   * the researcher wants to take a do-file Nora wrote and use it
   * elsewhere, or save it to their own filesystem.
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
    leftAction.innerHTML = iconSvg('copy');
    leftAction.addEventListener('click', (ev) => {
      ev.stopPropagation();
      copyImageToClipboard(f.data, f.mime || 'image/png', f.name);
    });
    primaryConfigured = true;
  } else if (kind === 'graph' && f.path) {
    leftAction.title = 'Open in default viewer';
    leftAction.setAttribute('aria-label', 'Open');
    leftAction.innerHTML = iconSvg('openExternal');
    leftAction.addEventListener('click', (ev) => {
      ev.stopPropagation();
      if (window.pywebview && window.pywebview.api &&
          typeof window.pywebview.api.open_path === 'function') {
        window.pywebview.api.open_path(f.path);
      }
    });
    primaryConfigured = true;
  } else if (kind === 'script' || kind === 'log') {
    leftAction.title = 'Copy file contents to clipboard';
    leftAction.setAttribute('aria-label', 'Copy contents');
    leftAction.innerHTML = iconSvg('copy');
    leftAction.addEventListener('click', (ev) => {
      ev.stopPropagation();
      // Pass the full path (not just f.name) so the bridge gate
      // can verify membership in the panel listing by absolute
      // path. The Files panel surfaces researcher-uploaded
      // scripts and logs (run-dir scripts are hidden by design
      // — they already render on the result card).
      copySessionFileText(f.path || f.name, f.name);
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
// the icon picks up the row's hover/focus colors. Single registry
// so the same glyph reads identically wherever it appears: the
// chat-bubble copy button and the Files-panel copy action are the
// SAME copy icon, the bubble's transient post-copy checkmark is the
// SAME check icon, etc. Add a new key here rather than inlining
// another <svg> string at a call site.
const ICON_SVG = {
  copy: (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
    '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
    '</svg>'
  ),
  edit: (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M12 20h9"></path>' +
    '<path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>' +
    '</svg>'
  ),
  // Heavier stroke (2.5 vs 2) so the transient post-copy checkmark
  // reads as confirmation rather than just another monochrome stroke.
  check: (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" ' +
    'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<polyline points="20 6 9 17 4 12"></polyline>' +
    '</svg>'
  ),
  // 16×16 viewBox for this one — the original "external link" glyph
  // was authored at 16px and the path coordinates reflect that.
  openExternal: (
    '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">' +
    '<path fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" ' +
    'd="M9 2h5v5M14 2L7 9M3 4h3M3 4v9h9v-3"/>' +
    '</svg>'
  ),
  // Pin glyphs — minimalist circle-head + needle. ``pin`` is the
  // outlined "click to pin" affordance (hollow head). ``pinFilled``
  // shares the same silhouette with a solid head so the engaged
  // state reads at a glance without needing an accent color: the
  // filled head plus the row's position at the top of the list is
  // enough to spot pinned sessions.
  pin: (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" ' +
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<circle cx="12" cy="8" r="3.5"></circle>' +
    '<line x1="12" y1="11.5" x2="12" y2="20"></line>' +
    '</svg>'
  ),
  pinFilled: (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" ' +
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<circle cx="12" cy="8" r="3.5" fill="currentColor"></circle>' +
    '<line x1="12" y1="11.5" x2="12" y2="20"></line>' +
    '</svg>'
  ),
};

function iconSvg(name) {
  return ICON_SVG[name] || '';
}

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

// ----- edit & rewind ----------------------------------------------------
//
// Researcher clicks the pencil on a user bubble → that bubble flips into
// edit mode (textarea + Run / Cancel). Run calls
// ``window.pywebview.api.rewind_to(turn_index)`` on the bridge — which
// truncates ``chat_history.jsonl`` at that user message, hides every
// dropped result row in the store, clears the runner's pending
// attachments, and resets the provider session. JS then calls
// ``replayHistory()`` to repaint the trimmed transcript and ``send_message``
// to fire the revised text as a fresh turn — events stream in normally
// on top.
//
// One bubble may be in edit mode at a time; entering edit on a second
// bubble cancels the first. Edit is refused while the session is busy
// (``busySessions.has(currentCwd)``); the researcher must Stop the
// running turn first, which the toast spells out.

let activeEditWrapper = null;

function enterEditMode(wrapper) {
  if (!wrapper) return;
  if (currentCwd && busySessions.has(currentCwd)) {
    toast(
      'Stop the running turn first, then click edit again.',
      'info',
    );
    return;
  }
  // Cancel any other in-progress edit. Two open editors would let a
  // researcher accidentally rewind through both, with the second
  // landing on a transcript that already started shifting under the
  // first.
  if (activeEditWrapper && activeEditWrapper !== wrapper) {
    cancelEditMode(activeEditWrapper);
  }
  if (wrapper.classList.contains('editing')) return;
  activeEditWrapper = wrapper;
  wrapper.classList.add('editing');

  const body = wrapper.querySelector('.message-body');
  if (!body) return;
  // Stash the original DOM so Cancel restores exactly what was
  // there (rendered markdown, attachments hidden by their own row,
  // etc.). textContent + innerHTML aren't enough because some
  // bubbles carry image rows above and chip rows below — those
  // siblings stay put; only the body swaps.
  wrapper.__noraEditOriginalBody = body;

  const editor = document.createElement('div');
  editor.className = 'message-edit-mode';
  const textarea = document.createElement('textarea');
  textarea.className = 'message-edit-textarea';
  textarea.value = wrapper.__noraRawText || '';
  textarea.rows = Math.min(12, Math.max(2, textarea.value.split('\n').length + 1));
  textarea.addEventListener('keydown', (e) => {
    // Submit on Enter (no modifier) and on Cmd/Ctrl+Enter — same
    // affordances as the composer. Shift+Enter inserts a newline.
    const isEnter = e.key === 'Enter';
    const submitPlain = isEnter && !e.shiftKey && !e.metaKey && !e.ctrlKey;
    const submitModifier = isEnter && (e.metaKey || e.ctrlKey) && !e.shiftKey;
    if (submitPlain || submitModifier) {
      e.preventDefault();
      runEditedMessage(wrapper);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      cancelEditMode(wrapper);
    }
  });

  const buttonRow = document.createElement('div');
  buttonRow.className = 'message-edit-buttons';
  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'message-edit-btn-cancel';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    cancelEditMode(wrapper);
  });
  const runBtn = document.createElement('button');
  runBtn.type = 'button';
  runBtn.className = 'message-edit-btn-run';
  runBtn.textContent = 'Run';
  runBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    runEditedMessage(wrapper);
  });
  buttonRow.appendChild(cancelBtn);
  buttonRow.appendChild(runBtn);

  editor.appendChild(textarea);
  editor.appendChild(buttonRow);

  // Replace the body in place with the editor; Cancel reverses this.
  body.replaceWith(editor);
  wrapper.__noraEditEditor = editor;
  textarea.focus();
  // Place the cursor at the end so the researcher can extend
  // immediately rather than overwriting.
  textarea.setSelectionRange(textarea.value.length, textarea.value.length);
}

function cancelEditMode(wrapper) {
  if (!wrapper || !wrapper.classList.contains('editing')) return;
  const editor = wrapper.__noraEditEditor;
  const original = wrapper.__noraEditOriginalBody;
  if (editor && original) {
    editor.replaceWith(original);
  }
  wrapper.__noraEditEditor = null;
  wrapper.__noraEditOriginalBody = null;
  wrapper.classList.remove('editing');
  if (activeEditWrapper === wrapper) activeEditWrapper = null;
}

function userMessageIndex(wrapper) {
  /* The bubble's 0-based position among ``.message.user`` bubbles in
   * DOM order — this matches the index of the corresponding
   * ``user_message`` event in ``chat_history.jsonl`` because the
   * persistence path appends one record per appended user bubble.
   * No counter to maintain; we just count siblings.
   */
  const all = Array.from(messagesEl.querySelectorAll('.message.user'));
  return all.indexOf(wrapper);
}

async function runEditedMessage(wrapper) {
  if (!wrapper) return;
  const editor = wrapper.__noraEditEditor;
  if (!editor) return;
  const textarea = editor.querySelector('.message-edit-textarea');
  if (!textarea) return;
  const newText = textarea.value.trim();
  if (!newText) {
    toast('Edited message is empty. Type something or click Cancel.', 'info');
    return;
  }
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.rewind_to !== 'function') {
    toast('Restart Nora to enable message edit.', 'info');
    return;
  }

  const turnIndex = userMessageIndex(wrapper);
  if (turnIndex < 0) {
    toast('Could not locate this message in the transcript.', 'error');
    return;
  }

  // Edit-and-rerun on a message that originally carried attachments
  // or images would silently drop them: the rewind path sends
  // ``newText`` only, and the bridge has no way to reconstitute
  // image bytes (they were never persisted, by design) or guarantee
  // the named scripts still exist on disk. The resulting "rerun"
  // is a different request than the original, while the original's
  // attachment evidence has just been truncated out of history.
  // Force the researcher to acknowledge the drop before proceeding;
  // the alternative ("Cancel and resend a new message") gives them
  // a clean way to redo it with fresh attachments.
  const attachmentChips = wrapper.querySelectorAll(
    '.message-attachment-chip',
  );
  const imageThumbs = wrapper.querySelectorAll('.message-image-thumb');
  if (attachmentChips.length > 0 || imageThumbs.length > 0) {
    const parts = [];
    if (attachmentChips.length > 0) {
      parts.push(
        attachmentChips.length === 1
          ? '1 attached file'
          : attachmentChips.length + ' attached files',
      );
    }
    if (imageThumbs.length > 0) {
      parts.push(
        imageThumbs.length === 1
          ? '1 image'
          : imageThumbs.length + ' images',
      );
    }
    const msg = (
      'This message originally included ' + parts.join(' and ') + '.\n\n'
      + 'Editing and re-running will send the new text WITHOUT those '
      + 'attachments. The rerun is therefore a different request than '
      + 'the original. To keep the attachments, cancel here and resend '
      + 'a new message instead.\n\n'
      + 'Continue without attachments?'
    );
    if (!window.confirm(msg)) {
      return;
    }
  }

  // Disable the editor's buttons so a double-click doesn't fire
  // the rewind twice. The composer / send button stays as-is —
  // the next user_message is what actually re-fires the turn, and
  // we want it to come from the live send path with the same
  // codepath any other message would.
  const buttons = editor.querySelectorAll('button');
  buttons.forEach((b) => { b.disabled = true; });

  // Which session this edit belongs to, captured before the rewind.
  // The rewind is slow, and everything after it (replay, send) acts
  // on the focused session — so without this the edit could be sent
  // into a different session.
  const editCwd = currentCwd;

  let res;
  try {
    res = await window.pywebview.api.rewind_to(turnIndex);
  } catch (err) {
    console.warn('rewind_to failed', err);
    toast(
      'Rewind failed: ' + (err && err.message ? err.message : err),
      'error',
    );
    buttons.forEach((b) => { b.disabled = false; });
    return;
  }
  if (!res || !res.ok) {
    const reason = (res && res.reason) || 'unknown';
    toast('Rewind refused: ' + reason, 'error');
    buttons.forEach((b) => { b.disabled = false; });
    return;
  }

  // Focus moved during the rewind. The rewind went to the right
  // session, but replay and send act on the focused one — so stop
  // here. The rewind already committed, so say so rather than
  // looking like nothing happened.
  const rewoundCwd = res.cwd || editCwd;
  if (rewoundCwd !== currentCwd) {
    toast(
      'Rewind applied to the previous session, but you switched away '
      + 'before it finished — the edited message was not sent. Switch '
      + 'back and resend it.',
      'info',
    );
    buttons.forEach((b) => { b.disabled = false; });
    activeEditWrapper = null;
    return;
  }

  // Rewind committed. Repaint the transcript from the truncated log
  // (which no longer includes this user message or anything after
  // it), THEN render the new user bubble, THEN fire the send. The
  // bubble has to be rendered JS-side here for the same reason the
  // composer does it on submit: the bridge persists ``user_message``
  // to ``chat_history.jsonl`` but does NOT dispatch a live
  // ``user_message`` event back to JS — there's no event to render
  // off of, so the composer always paints its own bubble. Without
  // this explicit ``appendUser`` call, the rewind path would leave
  // the researcher staring at a transcript that ends one message
  // before the one they just submitted, with no bubble for the
  // edit until ``assistant_text`` started streaming.
  activeEditWrapper = null;
  await replayHistory();

  // Render the new user bubble + busy state, mirroring the
  // composer's submit handler. The live turn carries the bubble
  // nodes so disposable-turn cleanup recognises it, and is keyed by
  // the cwd that is sending — same reason as the submit handler.
  const userEl = appendUser(newText, [], []);
  const rewindCwd = currentCwd;
  const localTurn = { id: null, nodes: [userEl], hasVisibleReply: false };
  setLiveTurn(rewindCwd, localTurn);
  setSending(true);

  if (typeof window.pywebview.api.send_message === 'function') {
    try {
      const turnId = await window.pywebview.api.send_message(newText);
      // Stop reads this id to mark the turn cancelled. Same
      // capture pattern as the composer's submit handler.
      if (typeof turnId === 'string' && turnId) {
        localTurn.id = turnId;
      }
    } catch (err) {
      console.warn('send_message after rewind failed', err);
      toast(
        'Rewind succeeded but send failed: '
        + (err && err.message ? err.message : err),
        'error',
      );
      // Roll back the busy state — the send didn't take, so the
      // composer should be ready to accept another attempt.
      setSending(false);
      clearLiveTurn(rewindCwd);
      return;
    }
  }

  if (res.hidden_count > 0) {
    toast(
      'Edited message. ' + res.hidden_count + ' prior result'
      + (res.hidden_count === 1 ? '' : 's')
      + ' hidden from model context.',
      'success',
    );
  }
}


async function copyMessageBubble(wrapper, btnEl) {
  /* Copy a chat bubble's text to the clipboard. Source is the
   * pre-render markdown / raw text the bubble was created with —
   * stored on ``wrapper.__noraRawText`` by ``append()``. We
   * deliberately don't read ``message-body.textContent`` because the
   * markdown renderer transforms inline math, smartens punctuation,
   * and collapses adjacent whitespace on lists, so a round-trip
   * through the DOM differs from the literal turn the model
   * generated or the user typed. Researchers pasting into another
   * chat or an editor want the source they sent / received.
   *
   * Visual feedback on success: swap the icon to a checkmark for
   * 1.4 s. The toast bus handles failure cases with the same
   * semantics as the Files-panel copy button.
   */
  if (!wrapper) return;
  const raw = wrapper.__noraRawText || '';
  if (!raw) {
    toast('Nothing to copy from this message.', 'info');
    return;
  }
  try {
    if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
      toast('Clipboard API unavailable in this WebView.', 'info');
      return;
    }
    await navigator.clipboard.writeText(raw);
  } catch (err) {
    console.warn('copyMessageBubble failed', err);
    toast('Copy failed: ' + (err && err.message ? err.message : err), 'error');
    return;
  }
  // Inline visual confirmation — a 1.4 s state swap on the button —
  // is less disruptive than a toast on every copy. The researcher
  // gets the receipt right where their cursor is. Toast is reserved
  // for failures, where attention DOES need to leave the bubble.
  if (btnEl) {
    const original = btnEl.innerHTML;
    btnEl.innerHTML = iconSvg('check');
    btnEl.classList.add('copied');
    setTimeout(() => {
      btnEl.innerHTML = original;
      btnEl.classList.remove('copied');
    }, 1400);
  }
}


async function copySessionFileText(path, displayName) {
  /* Pull a script's or log's text content from the bridge and
   * write it to the system clipboard so the researcher can paste
   * it into another chat or an external editor. Sister of
   * copyImageToClipboard for non-image kinds.
   *
   * Takes the full ``path`` (not just a name) so the bridge can
   * verify the row in the panel listing — the bridge enforces
   * the gate against the Files-panel enumeration, so passing a
   * disk path the panel never lists comes back as a refusal.
   * ``displayName`` is the row label shown in the toast.
   *
   * The bridge enforces the size cap and the script/log
   * extension allowlist; on the JS side we just relay the result
   * to the clipboard and surface the bridge's "reason" verbatim
   * if the read failed (so an over-the-cap log produces the
   * researcher-actionable hint about opening the folder, not a
   * generic "copy failed").
   */
  if (!path) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.read_session_file_text !== 'function') {
    toast('Restart Nora to enable copy-from-Files.', 'info');
    return;
  }
  let res;
  try {
    res = await window.pywebview.api.read_session_file_text(path);
  } catch (err) {
    console.warn('read_session_file_text failed', err);
    toast(
      'Copy failed: ' + (err && err.message ? err.message : err),
      'error',
    );
    return;
  }
  if (!res || !res.ok) {
    toast('Copy failed: ' + ((res && res.reason) || 'unknown'), 'error');
    return;
  }
  try {
    if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
      toast('Clipboard API unavailable in this WebView.', 'info');
      return;
    }
    await navigator.clipboard.writeText(res.text || '');
    toast('Copied ' + (displayName || res.name || path) + ' to clipboard.', 'success');
  } catch (err) {
    console.warn('clipboard.writeText failed', err);
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
    // Drop matching composer chips before re-rendering. ``res.unstaged``
    // is the authoritative list of staged names the backend just
    // dropped from the runner's pending_* lists — for run-dir scripts
    // it carries the label-derived display name (``linear_regression.py``)
    // that the chip shows, not the on-disk basename (``script.py``)
    // in ``res.name``. Splicing by both keeps the chip row honest
    // for older bridges that don't populate ``unstaged``.
    const dropped = Array.isArray(res.unstaged) ? res.unstaged.slice() : [];
    if (res.name) dropped.push(res.name);
    if (dropped.length > 0) {
      for (let i = stagedDataNotices.length - 1; i >= 0; i--) {
        if (dropped.includes(stagedDataNotices[i])) {
          stagedDataNotices.splice(i, 1);
        }
      }
    }
    renderAttachments();
  } catch (err) {
    console.warn('delete_session_file failed', err);
    toast('Could not delete: ' + (err && err.message ? err.message : err), 'error');
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
  // selection itself. Period separator matches the Files / Model
  // popups so the three top-bar chips read identically.
  header.innerHTML =
    '<strong>Permission</strong>. Ceiling on variable details ' +
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
// Effort ladders are PER PROVIDER — Anthropic offers max, OpenAI
// stops at xhigh — so the bar is rebuilt from the selected model's
// provider on every render. effortsByProvider holds all of them so a
// model switch repaints without another round-trip.
let effortsByProvider = {};    // {provider: [{id,label}]}
let currentEffortId = null;    // the backend's authoritative effort level
let defaultEffortId = null;    // catalog default (dotted in the bar)

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
    effortsByProvider = res.efforts_by_provider || {};
    currentEffortId = res.current_effort || null;
    defaultEffortId = res.default_effort || null;
    renderModelChip();
  } catch (err) {
    console.warn('list_models failed', err);
  }
}

function renderModelChip() {
  if (!modelChip || !modelChipLabel) return;
  const info = availableModels.find((m) => m.id === currentModelId);
  // "Sonnet 5 · xhigh" — the effort rides the chip as a dim mono
  // suffix so the current level is visible without opening the
  // popup (the same at-a-glance treatment Claude Code gives its
  // effort setting). Rendered as two spans so the suffix can be
  // styled independently; the label span is rebuilt each time.
  modelChipLabel.textContent = '';
  const nameSpan = document.createElement('span');
  nameSpan.textContent = info ? info.label : 'Model';
  modelChipLabel.appendChild(nameSpan);
  if (currentEffortId) {
    const effSpan = document.createElement('span');
    effSpan.className = 'model-chip-effort';
    effSpan.textContent = ' · ' + currentEffortId;
    modelChipLabel.appendChild(effSpan);
  }
  // Keep the context-chip ceiling in sync with the selected model
  // so the X / Y ratio reflects that model's actual window. If a
  // turn_done already painted the chip against the default window
  // (the loadModels / first-turn race in showChat), re-render so the
  // ratio reflects the now-known ceiling.
  if (info && info.context_window) {
    const changed = contextWindow !== info.context_window;
    contextWindow = info.context_window;
    // Window changed — re-render against the new denominator if we
    // have a solid count cached. No backend recount: same-tokenizer
    // model swap doesn't change the numerator, and cross-tokenizer
    // swap only matters once a turn fires under the new model
    // (which then triggers a recount on its own).
    if (changed && lastContextCount) {
      lastContextCount = { ...lastContextCount, ceiling: contextWindow };
      renderContextChip();
    }
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
  //     Sonnet 5 (1M)
  //     Opus 5 (1M)
  //     Fable 5 (1M)
  //   OpenAI
  //     GPT-5.6 Terra (1.05M)
  //     GPT-5.6 Sol (1.05M)
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

  // Effort bar — the same dial as Claude Code's effort picker,
  // embedded under the model list so "which model" and "how hard it
  // thinks" live in one place. The ladder belongs to the SELECTED
  // model's provider (Anthropic goes to max, OpenAI stops at xhigh),
  // so it's looked up per render and changes when the model does.
  // Segmented control; the active level is filled, and the default
  // carries a small dot so a researcher who wandered off it can find
  // the way back.
  const currentInfo = availableModels.find((m) => m.id === currentModelId);
  const currentProvider = (currentInfo && currentInfo.provider) || 'anthropic';
  const availableEfforts = effortsByProvider[currentProvider] || [];
  if (availableEfforts.length > 0) {
    const effHeader = document.createElement('div');
    effHeader.className = 'model-group-header';
    effHeader.textContent = 'Effort';
    wrap.appendChild(effHeader);

    const seg = document.createElement('div');
    seg.className = 'effort-seg';
    seg.setAttribute('role', 'radiogroup');
    seg.setAttribute('aria-label', 'Reasoning effort');
    availableEfforts.forEach((e) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'effort-seg-btn';
      btn.dataset.effort = e.id;
      btn.setAttribute('role', 'radio');
      const isActive = e.id === currentEffortId;
      btn.setAttribute('aria-checked', isActive ? 'true' : 'false');
      if (isActive) btn.classList.add('active');
      if (e.id === defaultEffortId) btn.classList.add('is-default');
      btn.textContent = e.id;
      btn.title = e.label + (e.id === defaultEffortId ? ' (default)' : '');
      btn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        if (e.id === currentEffortId) return;
        setEffort(e.id);
      });
      seg.appendChild(btn);
    });
    wrap.appendChild(seg);
  }

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

function showInstallConfirmationModal(evt) {
  /* Hard consent gate for the ``install_packages`` tool. The Python
   * handler is awaiting a response keyed by ``evt.token``; the
   * researcher's click on Approve / Deny calls
   * ``respond_install_confirmation`` to release that future. If the
   * researcher closes the modal without clicking either button
   * (Esc / overlay-click), we send an explicit deny so the handler
   * doesn't sit on its 5-minute timeout.
   *
   * Multiple concurrent requests are stacked: each call creates its
   * own overlay (token-keyed) so an early decision can't be applied
   * to a later request by accident. */
  if (!evt || !evt.token) return;

  const overlay = document.createElement('div');
  overlay.className = 'install-confirmation-overlay';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');
  overlay.setAttribute('aria-label', 'Confirm package install');
  overlay.tabIndex = -1;

  const card = document.createElement('div');
  card.className = 'install-confirmation-card';

  const title = document.createElement('div');
  title.className = 'install-confirmation-title';
  const actionLabel = (evt.action === 'remove')
    ? 'Remove'
    : (evt.action === 'reinstall') ? 'Reinstall' : 'Install';
  title.textContent = actionLabel + ' ' + (evt.language || '') + ' packages?';

  const desc = document.createElement('div');
  desc.className = 'install-confirmation-desc';
  desc.textContent = (
    'Nora wants to ' + actionLabel.toLowerCase() + ' the following '
    + 'packages on this machine. This runs the language’s package '
    + 'manager outside the sandbox and writes to your user library.'
  );

  const pkgList = document.createElement('ul');
  pkgList.className = 'install-confirmation-pkgs';
  (evt.packages || []).slice(0, 50).forEach((name) => {
    const li = document.createElement('li');
    li.textContent = String(name);
    pkgList.appendChild(li);
  });
  if ((evt.packages || []).length > 50) {
    const more = document.createElement('li');
    more.className = 'install-confirmation-more';
    more.textContent = '… and ' + ((evt.packages.length - 50)) + ' more';
    pkgList.appendChild(more);
  }

  const actions = document.createElement('div');
  actions.className = 'install-confirmation-actions';
  const denyBtn = document.createElement('button');
  denyBtn.type = 'button';
  denyBtn.className = 'install-confirmation-deny';
  denyBtn.textContent = 'Deny';
  const approveBtn = document.createElement('button');
  approveBtn.type = 'button';
  approveBtn.className = 'install-confirmation-approve';
  approveBtn.textContent = actionLabel;
  actions.appendChild(denyBtn);
  actions.appendChild(approveBtn);

  card.appendChild(title);
  card.appendChild(desc);
  card.appendChild(pkgList);
  card.appendChild(actions);
  overlay.appendChild(card);

  let resolved = false;
  const respond = async (approved) => {
    if (resolved) return;
    resolved = true;
    overlay.remove();
    document.removeEventListener('keydown', onKey);
    if (!window.pywebview || !window.pywebview.api) return;
    if (typeof window.pywebview.api.respond_install_confirmation !== 'function') {
      // Older bridge build without the new method. The Python
      // handler will time out; nothing we can do from JS to fix.
      return;
    }
    try {
      await window.pywebview.api.respond_install_confirmation(evt.token, !!approved);
    } catch (err) {
      console.warn('respond_install_confirmation failed', err);
    }
  };

  approveBtn.addEventListener('click', () => respond(true));
  denyBtn.addEventListener('click', () => respond(false));
  // Esc / overlay-click default to deny — closing without an explicit
  // approve is the safe interpretation, and Python is blocked on the
  // future, so silently dismissing would leave it waiting.
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) respond(false);
  });
  const onKey = (e) => {
    if (e.key === 'Escape') respond(false);
    // Enter MUST NOT unconditionally approve. The modal focuses
    // Deny by default (see ``denyBtn.focus()`` below), and an
    // unguarded ``Enter → respond(true)`` contradicted that — a
    // researcher hitting Enter on the focused Deny button got an
    // Approve anyway, which is the opposite of what every native
    // dialog convention promises. Only approve if Approve actually
    // has keyboard focus (the researcher Tabbed to it deliberately).
    // Native button activation on the focused element handles the
    // visual case correctly without this branch; the explicit gate
    // is here so a focus change that happens between events still
    // routes correctly.
    if (e.key === 'Enter' && document.activeElement === approveBtn) {
      respond(true);
    }
  };
  document.addEventListener('keydown', onKey);
  document.body.appendChild(overlay);
  // Focus Deny by default so an absent-minded Enter denies (the
  // button's native Enter handler clicks it); the researcher must
  // Tab to Approve before Enter approves.
  denyBtn.focus();
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
    const requestCwd = currentCwd;
    const res = await window.pywebview.api.set_model(modelId);
    // The swap is slow, so focus may have moved. The picker shows the
    // focused session, so painting another session's model here would
    // advertise the wrong one. The backend already applied it
    // correctly; we just skip the repaint.
    const stillFocused = (res && res.cwd)
      ? res.cwd === currentCwd
      : requestCwd === currentCwd;
    if (!res || !res.ok) {
      if (!silent) {
        const reason = res && res.reason ? res.reason : 'unknown';
        toast('Model switch failed. ' + sentenceCase(reason), 'error', 'model');
      }
      return;
    }
    if (!stillFocused) {
      // Landed in a session the researcher left. Say so, otherwise
      // the click looks like it did nothing.
      if (!silent && !res.unchanged) {
        toast('Model set to ' + (res.label || modelId) + ' for the previous session.', 'info', 'model');
      }
      return;
    }
    currentModelId = modelId;
    // A cross-provider switch can clamp the effort onto the new
    // provider's ladder (Anthropic max -> OpenAI xhigh). The backend
    // is authoritative about where it landed, so adopt what it
    // reports rather than assuming the level carried over.
    const clamped = res.effort && res.effort !== currentEffortId;
    if (res.effort) currentEffortId = res.effort;
    renderModelChip();
    if (!silent && !res.unchanged) {
      const tail = clamped
        ? ' Effort moved to ' + res.effort + ' (highest this provider offers).'
        : '';
      toast('Model switched to ' + (res.label || modelId) + '. Takes effect on the next message.' + tail, 'success', 'model');
    }
  } catch (err) {
    console.warn('set_model failed', err);
    if (!silent) toast('Model switch failed. ' + sentenceCase(err), 'error', 'model');
  }
}

async function setEffort(effortId) {
  /* Switch the focused session's reasoning effort. Mirrors setModel:
   * the backend is authoritative, so we only repaint after it says ok.
   * The popup stays open — effort is a dial researchers nudge and
   * compare, not a one-shot pick like the model — and the segmented
   * control re-renders in place. Anthropic sessions re-warm on the
   * next message (the Agent SDK takes effort at launch only), so the
   * toast says so when the backend flags it. */
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.set_effort !== 'function') {
    toast('Restart Nora to enable effort switching.', 'error', 'model');
    return;
  }
  try {
    const requestCwd = currentCwd;
    const res = await window.pywebview.api.set_effort(effortId);
    // Same focus race as setModel — don't paint another session's
    // setting onto the picker.
    const stillFocused = (res && res.cwd)
      ? res.cwd === currentCwd
      : requestCwd === currentCwd;
    if (!res || !res.ok) {
      const reason = res && res.reason ? res.reason : 'unknown';
      toast('Effort switch failed. ' + sentenceCase(reason), 'error', 'model');
      return;
    }
    if (!stillFocused) {
      if (!res.unchanged) {
        toast('Effort set to ' + (res.label || effortId) + ' for the previous session.', 'info', 'model');
      }
      return;
    }
    currentEffortId = effortId;
    renderModelChip();
    if (!res.unchanged) {
      const tail = res.conversation_rewarmed
        ? ' Session re-warms on the next message.'
        : ' Takes effect on the next message.';
      toast('Effort set to ' + (res.label || effortId) + '.' + tail, 'success', 'model');
    }
  } catch (err) {
    console.warn('set_effort failed', err);
    toast('Effort switch failed. ' + sentenceCase(err), 'error', 'model');
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
    // Each row: a pin toggle on the far left, the clickable session
    // body in the middle (switches cwd), rename + trash on the right.
    // Every action button is a sibling of the session-item button
    // rather than a nested child — putting interactives inside the
    // <button> is invalid nesting AND would make their clicks bubble
    // into the switch handler.
    const row = document.createElement('div');
    row.className = 'session-row';
    if (s.path === currentPath) row.classList.add('active');
    if (s.pinned) row.classList.add('pinned');
    row.title = s.path;

    // Pin toggle — leftmost. Two visually distinct icons make pinned
    // vs unpinned readable at a glance without reading the title
    // attribute, and the row is also tagged ``.pinned`` so we can
    // also hint with a subtle background/weight change in CSS.
    const pinBtn = document.createElement('button');
    pinBtn.type = 'button';
    pinBtn.className = 'session-pin';
    if (s.pinned) pinBtn.classList.add('is-pinned');
    pinBtn.setAttribute(
      'aria-label', s.pinned ? 'Unpin session' : 'Pin session to top'
    );
    pinBtn.title = s.pinned ? 'Unpin from top' : 'Pin to top';
    pinBtn.innerHTML = iconSvg(s.pinned ? 'pinFilled' : 'pin');
    pinBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleSessionPinned(s.path, !s.pinned);
    });
    row.appendChild(pinBtn);

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'session-item';
    // ``data-path`` is read by setSessionBusy() to flip the busy
    // dot on this row when a turn starts/ends — including for
    // background sessions the user isn't currently looking at.
    btn.dataset.path = s.path;
    if (busySessions.has(s.path)) btn.classList.add('busy');
    // Single-line layout: [title …] [age | busy-dot]
    // ``s.title`` already resolves to custom_name if the researcher
    // set one, otherwise to the dataset filename ("<first> +N" for
    // multi-file sessions), otherwise to a session-stamp fallback.
    // The busy dot lives in the same right-side slot as the age and
    // CSS flips visibility — pulse replaces "1d" while a turn runs.
    const titleEl = document.createElement('span');
    titleEl.className = 'session-title';
    titleEl.textContent = s.title || (s.datasets.length
      ? s.datasets.join(', ')
      : '(no data files)');
    btn.appendChild(titleEl);

    const ageEl = document.createElement('span');
    ageEl.className = 'session-age';
    ageEl.textContent = formatSessionAge(s.timestamp);
    btn.appendChild(ageEl);

    const dot = document.createElement('span');
    dot.className = 'session-busy-dot';
    dot.setAttribute('aria-hidden', 'true');
    btn.appendChild(dot);

    btn.addEventListener('click', () =>
      switchSession(s.path, s.path === currentPath)
    );
    row.appendChild(btn);

    // Folder-backed sessions (opened via the folder picker) are
    // not stored under ~/.nora-sessions/, so the backend's
    // delete_session and set_session_name reject them outright
    // (they require parent == SESSIONS_ROOT). Offering the
    // buttons here would prompt the researcher with a
    // destructive confirm or an editable name, then fail on
    // commit. Hide them so the controls match what the backend
    // will accept; the researcher manages their own project
    // dir for those sessions.
    if (s.kind !== 'folder') {
      const rename = document.createElement('button');
      rename.type = 'button';
      rename.className = 'session-rename';
      rename.setAttribute('aria-label', 'Rename session');
      rename.title = 'Rename this session';
      rename.textContent = '✎';
      rename.addEventListener('click', (e) => {
        e.stopPropagation();
        // Putting an <input> inside the <button.session-item>
        // would be invalid nesting (interactive in interactive),
        // so swap the whole button for an edit container. On
        // commit/cancel, beginRenameSession's loadSessions()
        // refresh redraws the row from server state.
        const editor = document.createElement('div');
        editor.className = 'session-item session-item-editing';
        editor.textContent = s.custom_name || s.title || '';
        row.replaceChild(editor, btn);
        beginRenameSession(editor, s.path);
      });
      row.appendChild(rename);

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
    }

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

async function toggleSessionPinned(path, pinned) {
  /* Flip a session's pin-to-top flag through the bridge. The bridge
   * stamps ``pinned_at`` on the pin transition so the sort within
   * the pinned group surfaces the most-recently-pinned row first,
   * matching the researcher's most likely "I just pinned this, where
   * did it go" expectation.
   *
   * On success we re-fetch the session list rather than mutating the
   * row in place: a pin reorders the panel, which is easier to do
   * correctly from a fresh list_sessions render than by splicing the
   * existing DOM. The round-trip is cheap (list_sessions is a
   * directory walk).
   */
  if (!path) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.set_session_pinned !== 'function') {
    toast('Restart Nora to enable session pinning.', 'info');
    return;
  }
  try {
    const res = await window.pywebview.api.set_session_pinned(path, pinned);
    if (!res || !res.ok) {
      toast(
        'Pin failed: ' + ((res && res.reason) || 'unknown'),
        'error',
      );
      return;
    }
  } catch (err) {
    console.warn('set_session_pinned failed', err);
    toast('Pin failed: ' + (err && err.message ? err.message : err), 'error');
    return;
  }
  if (typeof loadSessions === 'function') loadSessions();
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

function beginRenameSession(targetEl, path) {
  /* Swap the static label for an inline <input>, focused and
   * pre-filled with the current text. Save on Enter or blur,
   * cancel on Escape. Used by the topbar pill and by sidebar
   * rows — the call site passes whichever element should be
   * replaced for the duration of the edit.
   *
   * Empty / whitespace-only saves are intentional: they clear the
   * custom name and revert to the auto-derived label (single
   * dataset filename, "<first> +N", or session timestamp).
   */
  if (!targetEl || !path) return;
  if (targetEl.classList.contains('editing')) return;
  if (!window.pywebview || !window.pywebview.api) return;
  if (typeof window.pywebview.api.set_session_name !== 'function') {
    toast('Restart Nora to enable session renaming.', 'info');
    return;
  }
  const original = targetEl.textContent || '';
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'session-rename-input';
  input.value = original;
  input.maxLength = 120;
  input.setAttribute('aria-label', 'Session name');
  // Cache layout context so we can put the static text back.
  const placeholder = document.createElement('span');
  placeholder.className = 'session-rename-placeholder';
  targetEl.classList.add('editing');
  targetEl.replaceChildren(placeholder, input);
  input.focus();
  input.select();

  let done = false;
  const restore = (text) => {
    if (done) return;
    done = true;
    targetEl.classList.remove('editing');
    targetEl.replaceChildren();
    targetEl.textContent = text;
    // Sidebar rows enter edit mode by swapping the row's button for
    // a stand-in edit container — restoring that container's text
    // alone won't bring back the rename/delete buttons. Re-rendering
    // the sidebar from server state is harmless for the topbar case
    // and necessary for the sidebar case.
    if (typeof loadSessions === 'function') loadSessions();
  };
  const commit = async () => {
    if (done) return;
    const next = input.value;
    // Block accidental double-fires (blur after Enter).
    done = true;
    targetEl.classList.remove('editing');
    targetEl.replaceChildren();
    // Show what the user typed immediately so the edit feels
    // local; the bridge call below confirms or corrects it.
    targetEl.textContent = next.trim() || original;
    try {
      const res = await window.pywebview.api.set_session_name(path, next);
      if (!res || !res.ok) {
        targetEl.textContent = original;
        toast(
          'Could not rename: ' + ((res && res.reason) || 'unknown'),
          'error',
        );
        return;
      }
      // Server's resolved title is authoritative — it falls back
      // to the auto-derived label when the input was empty.
      if (res.title) targetEl.textContent = res.title;
      // Sidebar rows also display the title; refresh so they
      // stay in sync.
      if (typeof loadSessions === 'function') loadSessions();
    } catch (err) {
      console.warn('set_session_name failed', err);
      targetEl.textContent = original;
      toast('Rename failed: ' + (err && err.message ? err.message : err), 'error');
    }
  };

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      commit();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      restore(original);
    }
  });
  input.addEventListener('blur', () => {
    // Don't commit if Escape already restored.
    if (!done) commit();
  });
  // Stop click bubbling so a sidebar-row rename doesn't also
  // trigger the row's switch-session handler.
  input.addEventListener('click', (e) => e.stopPropagation());
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

function formatSessionAge(epochSeconds) {
  /* Compact "time since inception" label for the session list.
   * < 1 min : "now"
   * < 1 h   : "Xm"
   * < 1 d   : "Xh"
   * < 30 d  : "Xd"
   * else    : "Xmo"
   */
  if (!epochSeconds) return '';
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - epochSeconds));
  if (secs < 60) return 'now';
  if (secs < 3600) return Math.floor(secs / 60) + 'm';
  if (secs < 86400) return Math.floor(secs / 3600) + 'h';
  if (secs < 86400 * 30) return Math.floor(secs / 86400) + 'd';
  return Math.floor(secs / (86400 * 30)) + 'mo';
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
  // Clear the LEAVING session's backend pending lists. ``showChat``
  // wipes the JS-side staged composer state (image thumbs, data
  // notices, mention chips) on the way in to the new session — but
  // the runner's ``pending_*`` lists are per-cwd and survive a focus
  // switch. Without this call, a script attachment / @-mention
  // staged in A but never sent rides invisibly with the next plain
  // message in A: the UI shows no chip, the backend silently
  // inlines the file. Best-effort: a missing bridge method or a
  // failing RPC just means we keep the prior behaviour, not a
  // hard error.
  const leavingCwd = currentCwd;
  if (
    leavingCwd
    && leavingCwd !== path
    && typeof window.pywebview.api.clear_pending_for_session === 'function'
  ) {
    try {
      await window.pywebview.api.clear_pending_for_session(leavingCwd);
    } catch (err) {
      console.warn('clear_pending_for_session failed', err);
    }
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
    // Don't hijack typing keys when the user is inside the inline
    // rename input — Backspace would otherwise trigger the row's
    // delete confirm instead of erasing a character.
    if (
      focused.tagName === 'INPUT'
      || focused.tagName === 'TEXTAREA'
      || focused.isContentEditable
    ) {
      return;
    }
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

      // Refresh permission chip (new data files get default
      // policy) and the Files panel; sidebar size also updates. The
      // topbar pill shows the cwd path now, not the auto-derived
      // session title — uploads don't change cwd, so the pill
      // doesn't need a repaint.
      if (res.policy) updatePolicyChip(res.policy);
      refreshFilesChip();
      loadSessions();
      // Script attachments inflate the next request — recount so the
      // chip reflects them.
      triggerContextRecount('attachment-add');
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
    return 'Start a session first. Drop files or pick a folder from the landing screen.';
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

// Backend ``reason`` strings are lowercase fragments ("wait for the
// turn in flight to finish"). Notices read them as a second sentence
// after "... failed.", so lift the first letter.
function sentenceCase(text) {
  const s = String(text);
  return s.charAt(0).toUpperCase() + s.slice(1);
}

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

// ----- feedback modal ----------------------------------------------------

const feedbackBtn = document.getElementById('feedback-btn');
const feedbackOverlay = document.getElementById('feedback-overlay');
const feedbackCloseBtn = document.getElementById('feedback-close');
const feedbackCancelBtn = document.getElementById('feedback-cancel');
const feedbackSubmitBtn = document.getElementById('feedback-submit');
const feedbackSubjectEl = document.getElementById('feedback-subject');
const feedbackMessageEl = document.getElementById('feedback-message');
const feedbackEmailEl = document.getElementById('feedback-email');
const feedbackMessageCounterEl = document.getElementById(
  'feedback-message-counter',
);

// Must match the textarea's ``maxlength`` AND the backend cap in
// ``ui.send_feedback``. The backend rejects anything over this so
// the cap is enforced server-side too — the UI counter is the
// helpful affordance, the backend is the defence in depth.
const FEEDBACK_MESSAGE_CAP = 8000;

function updateFeedbackMessageCounter() {
  /* Live character count under the textarea. Switches to a warning
   * tone in the last ~10% and to an error tone at the cap, so the
   * researcher notices before submit. The counter element itself
   * is ``aria-live=polite`` so screen readers also get the cue. */
  if (!feedbackMessageEl || !feedbackMessageCounterEl) return;
  const len = (feedbackMessageEl.value || '').length;
  feedbackMessageCounterEl.textContent = `${len} / ${FEEDBACK_MESSAGE_CAP}`;
  feedbackMessageCounterEl.classList.toggle(
    'near-limit',
    len >= Math.floor(FEEDBACK_MESSAGE_CAP * 0.9) && len < FEEDBACK_MESSAGE_CAP,
  );
  feedbackMessageCounterEl.classList.toggle(
    'at-limit',
    len >= FEEDBACK_MESSAGE_CAP,
  );
}
if (feedbackMessageEl) {
  feedbackMessageEl.addEventListener('input', updateFeedbackMessageCounter);
}

function openFeedback() {
  if (!feedbackOverlay) return;
  feedbackOverlay.classList.remove('hidden');
  // Sync the counter to whatever's currently in the textarea
  // (e.g. a partially-typed message kept across opens).
  updateFeedbackMessageCounter();
  // Focus the subject line — that's the natural starting field
  // (write the headline, then expand into the body). Landing the
  // caret in the body box first forced the researcher to tab
  // backwards to fill in the subject.
  if (feedbackSubjectEl) {
    feedbackSubjectEl.focus();
  } else if (feedbackMessageEl) {
    feedbackMessageEl.focus();
  }
}

function closeFeedback() {
  if (!feedbackOverlay) return;
  feedbackOverlay.classList.add('hidden');
}

function resetFeedback() {
  if (feedbackSubjectEl) feedbackSubjectEl.value = '';
  if (feedbackMessageEl) feedbackMessageEl.value = '';
  if (feedbackEmailEl) feedbackEmailEl.value = '';
  updateFeedbackMessageCounter();
}

if (feedbackBtn) feedbackBtn.addEventListener('click', openFeedback);
if (feedbackCloseBtn) feedbackCloseBtn.addEventListener('click', closeFeedback);
if (feedbackCancelBtn) feedbackCancelBtn.addEventListener('click', closeFeedback);
if (feedbackOverlay) {
  feedbackOverlay.addEventListener('click', (e) => {
    if (e.target === feedbackOverlay) closeFeedback();
  });
  // Cmd+Enter (or Ctrl+Enter) submits without forcing the
  // researcher to mouse over to the button. Plain Enter still
  // adds a newline inside the textarea — same convention as the
  // chat composer. Scoped to the overlay so the binding only
  // fires while the modal is open.
  feedbackOverlay.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    if (!(e.metaKey || e.ctrlKey)) return;
    e.preventDefault();
    submitFeedback();
  });
}

async function submitFeedback() {
  if (!feedbackMessageEl) return;
  const message = (feedbackMessageEl.value || '').trim();
  if (!message) {
    toast('Write a quick note before sending.', 'info');
    feedbackMessageEl.focus();
    return;
  }
  if (!window.pywebview || !window.pywebview.api
      || typeof window.pywebview.api.send_feedback !== 'function') {
    toast('Restart Nora to enable feedback.', 'error');
    return;
  }
  const replyTo = (feedbackEmailEl && feedbackEmailEl.value || '').trim();
  const subject = (feedbackSubjectEl && feedbackSubjectEl.value || '').trim();
  if (feedbackSubmitBtn) {
    feedbackSubmitBtn.disabled = true;
    feedbackSubmitBtn.textContent = 'Sending…';
  }
  try {
    const res = await window.pywebview.api.send_feedback(
      message,
      replyTo || null,
      subject || null,
    );
    if (!res || !res.ok) {
      toast('Could not send: ' + ((res && res.reason) || 'unknown'), 'error');
      return;
    }
    toast('Feedback sent. Thanks!', 'success');
    resetFeedback();
    closeFeedback();
  } catch (err) {
    console.warn('send_feedback failed', err);
    toast('Could not send: ' + (err && err.message ? err.message : err), 'error');
  } finally {
    if (feedbackSubmitBtn) {
      feedbackSubmitBtn.disabled = false;
      feedbackSubmitBtn.textContent = 'Send feedback';
    }
  }
}

if (feedbackSubmitBtn) feedbackSubmitBtn.addEventListener('click', submitFeedback);

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
    if (feedbackOverlay && !feedbackOverlay.classList.contains('hidden')) {
      closeFeedback();
      return;
    }
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

// Rebuild ``busySessions`` from the bridge's view of which runners
// have a turn in flight. Called on every page boot — both initial
// load and after a hard reload (Cmd+Shift+R) — because
// ``busySessions`` is JS module-scope state that gets wiped on
// every navigation. Without this, refreshing during a turn drops
// the loading indicator AND the sidebar busy dot even though the
// backend is still streaming events.
//
// Logs to the console so the researcher can verify the seed path
// fired (DevTools → Console). If you see "seed: API method missing"
// after a refresh, nora was started with an older Python image
// that didn't have ``list_busy_sessions`` — fully quit and relaunch.
async function seedBusySessions() {
  if (!window.pywebview || !window.pywebview.api) {
    console.log('[nora] seed: pywebview bridge not ready yet');
    return;
  }
  if (typeof window.pywebview.api.list_busy_sessions !== 'function') {
    console.log(
      '[nora] seed: API method missing — bridge predates ' +
      'list_busy_sessions; fully restart nora to pick it up'
    );
    return;
  }
  try {
    const res = await window.pywebview.api.list_busy_sessions();
    if (!res || !res.ok || !Array.isArray(res.cwds)) {
      console.log('[nora] seed: bridge returned unexpected shape', res);
      return;
    }
    busySessions.clear();
    res.cwds.forEach((cwd) => busySessions.add(cwd));
    console.log(
      `[nora] seed: ${res.cwds.length} busy session(s) — `,
      res.cwds,
    );
  } catch (err) {
    console.warn('[nora] seed: list_busy_sessions failed', err);
  }
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
    // Seed the busy-set BEFORE branching into showChat / showLanding.
    // showChat → syncComposerToFocus reads busySessions to decide
    // whether to show the loading indicator + Stop button, and
    // loadSessions reads it to decide whether to paint the sidebar
    // dot. Both fire inside showChat, so the seed has to land first
    // or the first paint shows an idle UI for a busy session.
    await seedBusySessions();
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
