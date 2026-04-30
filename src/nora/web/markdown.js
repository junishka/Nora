/* Minimal markdown → HTML renderer for Nora's web UI.
 *
 * Covers the subset Claude actually emits in assistant text:
 *   - Paragraphs (double newline)
 *   - Headers (# / ## / ###)
 *   - Fenced code blocks (```lang ... ```)
 *   - Inline code (`...`)
 *   - Bold (**...**), italic (*...* or _..._)
 *   - Unordered lists (- or * at line start) and ordered lists (1.)
 *   - Blockquotes (> at line start)
 *   - Inline links ([text](url)) — HTTPS only
 *
 * Deliberately NOT covered: arbitrary HTML (always escaped), setext
 * headers, tables, reference-style links, images. If Claude starts
 * emitting those, we add them — meanwhile, escaping keeps the UI
 * safe against anything surprising.
 *
 * Written in-tree rather than importing `marked` or similar so we
 * don't pull executable JS from a CDN (which would break the
 * "nothing phones home" posture) and don't ship third-party code
 * we haven't reviewed.
 */

(function () {
  'use strict';

  function escapeHtml(s) {
    return s
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // Apply inline transformations to a single-line or paragraph-of-text
  // HTML-escaped string.
  function renderInline(s) {
    // Inline code first — content inside backticks is literal and must
    // not be interpreted as bold/italic/etc.
    s = s.replace(/`([^`]+)`/g, function (_, code) {
      return '<code>' + escapeHtml(code) + '</code>';
    });
    // Bold before italic so **x** doesn't match as *_x_*.
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Italic with `*`: opening `*` must sit at a word boundary AND
    // not be followed by whitespace; closing `*` must not be
    // preceded by whitespace AND not be followed by a word char.
    // Without this, a bare `*` used as a math / code operator
    // (e.g. "x * y", "max(charity_age * 0.5)") opens an emphasis
    // that runs to the next `*` and italicises every sentence
    // between, a real failure mode when the model writes
    // pseudo-Stata / pseudo-pandas in prose.
    s = s.replace(
      /(^|[^A-Za-z0-9*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![A-Za-z0-9*])/g,
      '$1<em>$2</em>'
    );
    // Italic with `_`: same word-boundary rule. CommonMark explicitly
    // forbids intra-word `_` emphasis ("Cat_Dog_" is literal text,
    // not "Cat<em>Dog</em>"). Without this rule, identifiers like
    // `fp_dur_resolved` / `age_at_arrival` / `webal_new` get their
    // middle chunk italicised whenever the model mentions them in
    // prose.
    s = s.replace(
      /(^|[^A-Za-z0-9_])_(?!\s)([^_\n]+?)(?<!\s)_(?![A-Za-z0-9_])/g,
      '$1<em>$2</em>'
    );
    // Links — HTTPS only. Anything else falls through as plain text.
    s = s.replace(
      /\[([^\]]+)\]\((https:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>'
    );
    return s;
  }

  function render(text) {
    if (!text) return '';
    // Escape everything first — inline renderers operate on escaped
    // text but re-inject specific HTML tags they generate. This is
    // deliberately conservative.
    const escaped = escapeHtml(String(text));
    const lines = escaped.split('\n');
    const out = [];

    // State machine: walk lines, open/close blocks as needed.
    let i = 0;
    let inCodeBlock = false;
    let codeLines = [];
    let codeLang = '';
    let listType = null;     // 'ul' | 'ol' | null
    let listItems = [];      // pending <li> content
    let pendingBlankInList = false;  // see "Blank line" handling below
    let paraLines = [];

    // GitHub-flavored markdown pipe tables — the shape Claude emits
    // when it hands back a regression coefficient table:
    //   | Term | Coef. | SE |
    //   |------|-------|-----|
    //   | x    |  0.42 | 0.01 |
    function splitCells(row) {
      // Strip outer pipes, split on inner pipes, trim each cell.
      const inner = row.replace(/^\s*\|/, '').replace(/\|\s*$/, '');
      return inner.split('|').map((c) => c.trim());
    }
    function isTableSeparator(row) {
      // The separator row — `|---|---|---|` (with optional colons for
      // alignment, which we ignore for v1 — everything is left-
      // aligned). Allow extra spaces and minimum of one dash per
      // cell.
      if (!/^\s*\|/.test(row) || !/\|\s*$/.test(row)) return false;
      const cells = splitCells(row);
      return cells.length >= 1
        && cells.every((c) => /^:?-{3,}:?$/.test(c));
    }

    function flushPara() {
      if (paraLines.length === 0) return;
      const body = paraLines
        .map((ln) => renderInline(ln))
        .join('<br>');
      out.push('<p>' + body + '</p>');
      paraLines = [];
    }

    function flushList() {
      if (listType === null) return;
      const tag = listType;
      out.push('<' + tag + '>');
      listItems.forEach((item) => {
        out.push('<li>' + renderInline(item) + '</li>');
      });
      out.push('</' + tag + '>');
      listType = null;
      listItems = [];
      pendingBlankInList = false;
    }

    function flushCode() {
      if (!inCodeBlock) return;
      const body = codeLines.join('\n');
      const cls = codeLang ? ' class="lang-' + escapeHtml(codeLang) + '"' : '';
      // Code inside fenced blocks is already escaped (we escaped the
      // entire text up front), so we don't re-escape here.
      out.push('<pre><code' + cls + '>' + body + '</code></pre>');
      codeLines = [];
      codeLang = '';
      inCodeBlock = false;
    }

    for (i = 0; i < lines.length; i++) {
      const raw = lines[i];
      // Fenced code block toggle. Matches escaped ``` (the backticks
      // survive HTML-escape unchanged).
      const fence = raw.match(/^```(.*)$/);
      if (fence) {
        if (inCodeBlock) {
          flushCode();
        } else {
          flushPara();
          flushList();
          inCodeBlock = true;
          codeLang = fence[1].trim();
        }
        continue;
      }
      if (inCodeBlock) {
        codeLines.push(raw);
        continue;
      }

      // Blank line ends the current paragraph, but doesn't
      // immediately close an open list. Claude often emits loose
      // lists with a blank line between items (`1. foo\n\n1. bar`);
      // closing the `<ol>` at every blank means each item becomes
      // its own single-item list that restarts numbering at 1. We
      // instead set a flag and let the next non-blank line decide:
      // if it's another list item, we continue the list; otherwise
      // we flush.
      if (!raw.trim()) {
        flushPara();
        if (listType !== null) {
          pendingBlankInList = true;
        }
        continue;
      }

      // Table detection: a header row (starts and ends with `|`),
      // followed by a separator row (`|---|---|---|`). Consume all
      // subsequent pipe-rows as body rows until a non-table line.
      if (/^\s*\|.*\|\s*$/.test(raw) && i + 1 < lines.length
          && isTableSeparator(lines[i + 1])) {
        flushPara();
        flushList();
        const headerCells = splitCells(raw);
        const rows = [];
        i += 2;  // skip the header + the separator
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
          rows.push(splitCells(lines[i]));
          i++;
        }
        // Step back so the outer loop's `i++` doesn't skip the
        // line that ended the table.
        i--;
        // Wrap each table in a scroll container so CSS can put the
        // horizontal scrollbar on the wrapper instead of leaking it
        // up through the message body. Plain ``<table>`` outputs let
        // the page itself decide where to put the scrollbar; on a
        // wide regression matrix that produces an awkward double-
        // scrollbar (top and bottom of the message-body's overflow
        // area). The wrapper centralizes styling on ``.md-table``.
        const parts = ['<div class="md-table"><table>', '<thead><tr>'];
        headerCells.forEach((c) =>
          parts.push('<th>' + renderInline(c) + '</th>')
        );
        parts.push('</tr></thead>');
        if (rows.length) {
          parts.push('<tbody>');
          rows.forEach((row) => {
            parts.push('<tr>');
            row.forEach((c) =>
              parts.push('<td>' + renderInline(c) + '</td>')
            );
            parts.push('</tr>');
          });
          parts.push('</tbody>');
        }
        parts.push('</table></div>');
        out.push(parts.join(''));
        continue;
      }

      // Headings.
      const h = raw.match(/^(#{1,3})\s+(.*)$/);
      if (h) {
        flushPara();
        flushList();
        const level = h[1].length;
        out.push('<h' + level + '>' + renderInline(h[2]) + '</h' + level + '>');
        continue;
      }

      // Blockquote.
      const bq = raw.match(/^>\s?(.*)$/);
      if (bq) {
        flushPara();
        flushList();
        out.push('<blockquote>' + renderInline(bq[1]) + '</blockquote>');
        continue;
      }

      // Unordered list item.
      const ul = raw.match(/^\s*[-*]\s+(.*)$/);
      if (ul) {
        flushPara();
        if (listType !== 'ul') flushList();
        listType = 'ul';
        listItems.push(ul[1]);
        pendingBlankInList = false;
        continue;
      }

      // Ordered list item.
      const ol = raw.match(/^\s*\d+\.\s+(.*)$/);
      if (ol) {
        flushPara();
        if (listType !== 'ol') flushList();
        listType = 'ol';
        listItems.push(ol[1]);
        pendingBlankInList = false;
        continue;
      }

      // Otherwise accumulate into the current paragraph. If a blank
      // line separated us from an open list and this line isn't a
      // list item, the list is done — close it before starting the
      // paragraph.
      flushList();
      paraLines.push(raw);
    }

    // End-of-input flush.
    flushCode();
    flushPara();
    flushList();

    return out.join('\n');
  }

  // Expose on `window` so app.js can call NoraMarkdown.render(text).
  window.NoraMarkdown = { render: render };
})();
