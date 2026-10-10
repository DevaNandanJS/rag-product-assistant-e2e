/**
 * Filumart Assistant - Render Engine
 * Safe DOM construction, progressive markdown rendering, and debug views.
 * Strictly adheres to Rule 2.3: No innerHTML for untrusted data.
 * DOMPurify.sanitize(marked.parse()) is the ONLY innerHTML pathway.
 */

/**
 * Render markdown safely using marked and DOMPurify.
 * Falls back to plain text if marked or DOMPurify is not yet loaded.
 * 
 * @param {string} mdText Raw markdown string
 * @returns {string} Sanitized HTML string
 */
export function renderMarkdown(mdText) {
  if (!mdText) return '';

  // Check if marked and DOMPurify are available globally
  const hasMarked = typeof window !== 'undefined' && typeof window.marked !== 'undefined' && typeof window.marked.parse === 'function';
  const hasDOMPurify = typeof window !== 'undefined' && typeof window.DOMPurify !== 'undefined' && typeof window.DOMPurify.sanitize === 'function';

  if (!hasMarked || !hasDOMPurify) {
    // Safe text fallback
    const div = document.createElement('div');
    div.textContent = mdText;
    return `<p>${div.innerHTML.replace(/\n/g, '<br>')}</p>`;
  }

  // Pre-process citations: [S1] -> citation badge markdown/html
  const processed = mdText.replace(/\[(S\d+)\]/g, '<span class="cite-badge" data-source-id="$1">$1</span>');

  const rawHtml = window.marked.parse(processed, {
    gfm: true,
    breaks: true,
  });

  return window.DOMPurify.sanitize(rawHtml, {
    ADD_TAGS: ['span', 'button'],
    ADD_ATTR: ['data-source-id'],
  });
}

/**
 * Append a user message bubble to the chat thread.
 * 
 * @param {HTMLElement} threadElement The container element (#message-thread)
 * @param {string} text The user's query text
 * @returns {HTMLElement} The created row element
 */
export function appendUserMessage(threadElement, text) {
  const row = document.createElement('div');
  row.className = 'message-row message-user';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = text; // Plain text only - zero XSS possible

  row.appendChild(bubble);
  threadElement.appendChild(row);
  scrollToBottom(threadElement);
  return row;
}

/**
 * Append an assistant message placeholder to the chat thread.
 * Returns an object with methods to stream tokens and finalize markdown.
 * 
 * @param {HTMLElement} threadElement The container element (#message-thread)
 * @returns {{row: HTMLElement, bubble: HTMLElement, updateToken: Function, finalize: Function, setSources: Function}}
 */
export function appendAssistantMessage(threadElement) {
  const row = document.createElement('div');
  row.className = 'message-row message-assistant';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble streaming-cursor';
  row.appendChild(bubble);

  const sourcesContainer = document.createElement('div');
  sourcesContainer.className = 'sources-container';
  sourcesContainer.style.display = 'none'; // Hidden until sources arrive
  row.appendChild(sourcesContainer);

  threadElement.appendChild(row);
  scrollToBottom(threadElement);

  let accumulatedText = '';

  return {
    row,
    bubble,
    sourcesContainer,
    /**
     * Called on each incoming token chunk.
     */
    updateToken(token) {
      accumulatedText += token;
      // Progressive display: sanitize & render markdown so user sees formatted output
      bubble.innerHTML = renderMarkdown(accumulatedText);
      attachCitationListeners(bubble, sourcesContainer);
      scrollToBottom(threadElement);
    },
    /**
     * Called when stream completes.
     */
    finalize(fullText = null) {
      if (fullText !== null) {
        accumulatedText = fullText;
      }
      bubble.classList.remove('streaming-cursor');
      bubble.innerHTML = renderMarkdown(accumulatedText);
      attachCitationListeners(bubble, sourcesContainer);
      scrollToBottom(threadElement);
    },
    /**
     * Render source citation cards.
     */
    setSources(sources) {
      if (!sources || sources.length === 0) return;
      sourcesContainer.innerHTML = ''; // safe to clear
      sourcesContainer.style.display = 'flex';

      const heading = document.createElement('div');
      heading.className = 'sources-heading';
      heading.textContent = `Cited Sources (${sources.length})`;
      sourcesContainer.appendChild(heading);

      sources.forEach((src, idx) => {
        const card = buildSourceCard(src, idx + 1);
        sourcesContainer.appendChild(card);
      });

      attachCitationListeners(bubble, sourcesContainer);
      scrollToBottom(threadElement);
    },
  };
}

/**
 * Build a single source card element using safe DOM methods.
 * Strictly avoids innerHTML for any untrusted source fields.
 * 
 * @param {Object} source Source metadata object
 * @param {number} index 1-based index (for S1, S2, etc.)
 * @returns {HTMLElement} The card element
 */
export function buildSourceCard(source, index) {
  const card = document.createElement('div');
  card.className = 'source-card';
  const sourceId = source.id || `S${index}`;
  card.id = `source-card-${sourceId}`;
  card.setAttribute('data-source-id', sourceId);

  // Card Header
  const header = document.createElement('div');
  header.className = 'source-card-header';

  const badge = document.createElement('span');
  badge.className = 'source-id-badge';
  badge.textContent = sourceId;
  header.appendChild(badge);

  const title = document.createElement('span');
  title.className = 'source-title';
  title.textContent = source.product_name || source.product_id || 'Industrial Reference';
  header.appendChild(title);

  card.appendChild(header);

  // Card Metadata row
  const metaRow = document.createElement('div');
  metaRow.className = 'source-meta';

  if (source.document_id) {
    const docSpan = document.createElement('span');
    docSpan.textContent = `Doc: ${source.document_id}`;
    metaRow.appendChild(docSpan);
  }

  if (source.page_number !== undefined && source.page_number !== null) {
    const pageSpan = document.createElement('span');
    pageSpan.textContent = `Page ${source.page_number}`;
    metaRow.appendChild(pageSpan);
  }

  if (source.chunk_id) {
    const chunkSpan = document.createElement('span');
    chunkSpan.textContent = `Chunk: ${source.chunk_id}`;
    metaRow.appendChild(chunkSpan);
  }

  if (source.is_ocr) {
    const ocrSpan = document.createElement('span');
    ocrSpan.className = 'badge badge-loading';
    ocrSpan.style.fontSize = '10px';
    ocrSpan.style.padding = '2px 6px';
    const conf = source.ocr_confidence !== undefined ? ` (${Math.round(source.ocr_confidence * 100)}%)` : '';
    ocrSpan.textContent = `OCR${conf}`;
    metaRow.appendChild(ocrSpan);
  }

  card.appendChild(metaRow);

  // Snippet
  if (source.snippet) {
    const snippetDiv = document.createElement('div');
    snippetDiv.className = 'source-snippet';
    snippetDiv.textContent = `"${source.snippet.trim()}"`;
    card.appendChild(snippetDiv);
  }

  return card;
}

/**
 * Attach click listeners to citation badges inside assistant bubble
 * to smoothly highlight the corresponding source card.
 */
function attachCitationListeners(bubble, sourcesContainer) {
  const badges = bubble.querySelectorAll('.cite-badge');
  badges.forEach(badge => {
    badge.onclick = (e) => {
      e.preventDefault();
      const targetId = badge.getAttribute('data-source-id');
      if (!targetId) return;

      // Unhighlight others
      sourcesContainer.querySelectorAll('.source-card').forEach(c => {
        c.classList.remove('highlighted');
      });

      const targetCard = sourcesContainer.querySelector(`#source-card-${targetId}`) 
        || sourcesContainer.querySelector(`[data-source-id="${targetId}"]`);

      if (targetCard) {
        targetCard.classList.add('highlighted');
        targetCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    };
  });
}

/**
 * Update the retrieval debug drawer with structured inspection data.
 * Pure DOM construction - no innerHTML for untrusted contents.
 * 
 * @param {Object} debugData The debug payload from the backend
 */
export function renderDebugDrawer(debugData) {
  if (!debugData) return;

  // Metadata Section
  const metaBody = document.getElementById('debug-meta-body');
  if (metaBody && debugData.meta) {
    metaBody.innerHTML = '';
    const dl = document.createElement('div');
    for (const [key, val] of Object.entries(debugData.meta)) {
      const row = document.createElement('div');
      row.style.marginBottom = '4px';
      const kSpan = document.createElement('strong');
      kSpan.textContent = `${key}: `;
      const vSpan = document.createElement('span');
      vSpan.textContent = String(val);
      row.appendChild(kSpan);
      row.appendChild(vSpan);
      dl.appendChild(row);
    }
    metaBody.appendChild(dl);
  }

  // Gate / Router Section
  const gateBody = document.getElementById('debug-gate-body');
  if (gateBody && debugData.gate) {
    gateBody.innerHTML = '';
    const table = document.createElement('table');
    table.className = 'debug-table';
    for (const [key, val] of Object.entries(debugData.gate)) {
      const tr = document.createElement('tr');
      const td1 = document.createElement('td');
      td1.textContent = key;
      const td2 = document.createElement('td');
      td2.textContent = typeof val === 'object' ? JSON.stringify(val) : String(val);
      tr.appendChild(td1);
      tr.appendChild(td2);
      table.appendChild(tr);
    }
    gateBody.appendChild(table);
  }

  // Fused RRF Candidates Section
  const fusedBody = document.getElementById('debug-fused-body');
  if (fusedBody && Array.isArray(debugData.fused_candidates)) {
    fusedBody.innerHTML = '';
    if (debugData.fused_candidates.length === 0) {
      fusedBody.textContent = 'Zero fused candidates.';
    } else {
      const table = document.createElement('table');
      table.className = 'debug-table';
      const thead = document.createElement('tr');
      ['Rank', 'Chunk', 'Product', 'RRF Score'].forEach(h => {
        const th = document.createElement('th');
        th.textContent = h;
        thead.appendChild(th);
      });
      table.appendChild(thead);

      debugData.fused_candidates.forEach((cand, i) => {
        const tr = document.createElement('tr');
        const rTd = document.createElement('td');
        rTd.textContent = String(i + 1);
        const cTd = document.createElement('td');
        cTd.textContent = cand.chunk_id || cand.id || '-';
        const pTd = document.createElement('td');
        pTd.textContent = cand.product_id || '-';
        const sTd = document.createElement('td');
        sTd.textContent = cand.score !== undefined ? Number(cand.score).toFixed(4) : '-';
        tr.append(rTd, cTd, pTd, sTd);
        table.appendChild(tr);
      });
      fusedBody.appendChild(table);
    }
  }

  // Dense Ranks Section
  const denseBody = document.getElementById('debug-dense-body');
  if (denseBody && Array.isArray(debugData.dense_ranks)) {
    denseBody.innerHTML = '';
    renderRankList(denseBody, debugData.dense_ranks);
  }

  // Sparse Ranks Section
  const sparseBody = document.getElementById('debug-sparse-body');
  if (sparseBody && Array.isArray(debugData.sparse_ranks)) {
    sparseBody.innerHTML = '';
    renderRankList(sparseBody, debugData.sparse_ranks);
  }
}

function renderRankList(container, ranks) {
  if (ranks.length === 0) {
    container.textContent = 'None';
    return;
  }
  const table = document.createElement('table');
  table.className = 'debug-table';
  const thead = document.createElement('tr');
  ['#', 'Chunk ID', 'Score'].forEach(h => {
    const th = document.createElement('th');
    th.textContent = h;
    thead.appendChild(th);
  });
  table.appendChild(thead);

  ranks.forEach((r, idx) => {
    const tr = document.createElement('tr');
    const td1 = document.createElement('td');
    td1.textContent = String(idx + 1);
    const td2 = document.createElement('td');
    td2.textContent = r.chunk_id || r.id || '-';
    const td3 = document.createElement('td');
    td3.textContent = r.score !== undefined ? Number(r.score).toFixed(4) : '-';
    tr.append(td1, td2, td3);
    table.appendChild(tr);
  });
  container.appendChild(table);
}

/**
 * Update the header status badge.
 * 
 * @param {HTMLElement} badgeEl 
 * @param {string} state 'ok' | 'degraded' | 'loading' | 'error'
 * @param {string} text Text to display
 */
export function updateStatusBadge(badgeEl, state, text) {
  if (!badgeEl) return;
  badgeEl.className = `badge badge-${state}`;
  badgeEl.textContent = text;
}

/**
 * Scroll thread to bottom smoothly.
 */
export function scrollToBottom(threadElement) {
  if (!threadElement) return;
  threadElement.scrollTop = threadElement.scrollHeight;
}

// Global attachment for fallback script loading
if (typeof window !== 'undefined') {
  window.FiluRender = {
    renderMarkdown,
    appendUserMessage,
    appendAssistantMessage,
    buildSourceCard,
    renderDebugDrawer,
    updateStatusBadge,
    scrollToBottom,
  };
}
