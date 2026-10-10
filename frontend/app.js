/**
 * Filumart Assistant - Main Application Controller
 * Manages UI state machine, filter population, query dispatch, and SSE streams.
 */

// Import modules with fallback to window globals
import { fetchFilters, fetchHealth, streamAsk } from './api.js';
import { 
  appendUserMessage, 
  appendAssistantMessage, 
  renderDebugDrawer, 
  updateStatusBadge 
} from './render.js';

(function () {
  // DOM Element References
  const queryInput = document.getElementById('query-input');
  const btnSend = document.getElementById('btn-send');
  const btnStop = document.getElementById('btn-stop');
  const btnNewChat = document.getElementById('btn-new-chat');
  const btnDebugToggle = document.getElementById('btn-debug-toggle');
  const btnCloseDebug = document.getElementById('btn-close-debug');
  const btnResetFilters = document.getElementById('btn-reset-filters');
  const messageThread = document.getElementById('message-thread');
  const statusBar = document.getElementById('status-bar');
  const statusBadge = document.getElementById('status-badge');
  const debugDrawer = document.getElementById('debug-drawer');
  const welcomeCard = document.getElementById('welcome-card');

  // Filter Select Elements
  const filterCategory = document.getElementById('filter-category');
  const filterSupplier = document.getElementById('filter-supplier');
  const filterCountry = document.getElementById('filter-country');
  const filterProduct = document.getElementById('filter-product');

  // App State
  let currentAbortController = null;
  let isGenerating = false;

  // Resolve API/Render methods (module or global)
  const api = {
    fetchFilters: typeof fetchFilters === 'function' ? fetchFilters : window.FiluAPI?.fetchFilters,
    fetchHealth: typeof fetchHealth === 'function' ? fetchHealth : window.FiluAPI?.fetchHealth,
    streamAsk: typeof streamAsk === 'function' ? streamAsk : window.FiluAPI?.streamAsk,
  };

  const render = {
    appendUserMessage: typeof appendUserMessage === 'function' ? appendUserMessage : window.FiluRender?.appendUserMessage,
    appendAssistantMessage: typeof appendAssistantMessage === 'function' ? appendAssistantMessage : window.FiluRender?.appendAssistantMessage,
    renderDebugDrawer: typeof renderDebugDrawer === 'function' ? renderDebugDrawer : window.FiluRender?.renderDebugDrawer,
    updateStatusBadge: typeof updateStatusBadge === 'function' ? updateStatusBadge : window.FiluRender?.updateStatusBadge,
  };

  /**
   * Initialize App: load health status & catalog filters.
   */
  async function init() {
    setupEventListeners();
    await checkHealth();
    await loadFilters();
  }

  /**
   * Health Check: Query GET /health and set badge.
   */
  async function checkHealth() {
    render.updateStatusBadge(statusBadge, 'loading', 'Connecting…');
    try {
      const data = await api.fetchHealth();
      if (data.status === 'ok') {
        render.updateStatusBadge(statusBadge, 'ok', 'Online (Ready)');
      } else if (data.status === 'degraded') {
        const mode = data.llm_mode || 'retrieval-only';
        render.updateStatusBadge(statusBadge, 'degraded', `Degraded (${mode})`);
      } else {
        render.updateStatusBadge(statusBadge, 'error', `Status: ${data.status}`);
      }
    } catch (err) {
      render.updateStatusBadge(statusBadge, 'error', 'Offline');
      console.warn('Health check failed:', err);
    }
  }

  /**
   * Load catalog filters from GET /filters and populate dropdowns.
   */
  async function loadFilters() {
    try {
      const data = await api.fetchFilters();
      populateSelect(filterCategory, data.categories, 'All Categories');
      populateSelect(filterSupplier, data.suppliers, 'All Suppliers');
      populateSelect(filterCountry, data.countries, 'All Countries');
      populateSelect(filterProduct, data.products, 'All Products');
    } catch (err) {
      console.warn('Failed to load catalog filters:', err);
    }
  }

  function populateSelect(selectEl, items, defaultLabel) {
    if (!selectEl || !Array.isArray(items)) return;
    selectEl.innerHTML = '';
    const defOpt = document.createElement('option');
    defOpt.value = '';
    defOpt.textContent = defaultLabel;
    selectEl.appendChild(defOpt);

    items.forEach(item => {
      if (!item) return;
      const opt = document.createElement('option');
      opt.value = item;
      opt.textContent = item;
      selectEl.appendChild(opt);
    });
  }

  /**
   * Set up UI event listeners.
   */
  function setupEventListeners() {
    // Send button & Enter key
    btnSend?.addEventListener('click', handleSend);
    queryInput?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });

    // Stop button
    btnStop?.addEventListener('click', handleStop);

    // New Chat
    btnNewChat?.addEventListener('click', handleNewChat);

    // Reset filters
    btnResetFilters?.addEventListener('click', () => {
      if (filterCategory) filterCategory.value = '';
      if (filterSupplier) filterSupplier.value = '';
      if (filterCountry) filterCountry.value = '';
      if (filterProduct) filterProduct.value = '';
    });

    // Debug Drawer Toggle
    btnDebugToggle?.addEventListener('click', () => {
      debugDrawer?.classList.toggle('drawer-hidden');
    });

    btnCloseDebug?.addEventListener('click', () => {
      debugDrawer?.classList.add('drawer-hidden');
    });

    // Prompt Chips in welcome card
    document.querySelectorAll('.prompt-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const query = chip.getAttribute('data-query');
        if (query && queryInput) {
          queryInput.value = query;
          handleSend();
        }
      });
    });
  }

  /**
   * Handle user query submission and initiate SSE stream.
   */
  async function handleSend() {
    if (isGenerating) return;

    const query = (queryInput?.value || '').trim();
    if (!query) {
      queryInput?.focus();
      return;
    }

    // Hide welcome card if present
    if (welcomeCard && welcomeCard.parentElement) {
      welcomeCard.remove();
    }

    // Collect active filters
    const filters = {};
    if (filterCategory?.value) filters.category = filterCategory.value;
    if (filterSupplier?.value) filters.supplier_name = filterSupplier.value;
    if (filterCountry?.value) filters.country = filterCountry.value;
    if (filterProduct?.value) filters.product_id = filterProduct.value;

    const payload = {
      question: query,
      filters: Object.keys(filters).length > 0 ? filters : null,
      debug: true,
      top_k: 5,
    };

    // Render user message
    render.appendUserMessage(messageThread, query);
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // Enter generating state
    setGeneratingState(true);
    if (statusBar) statusBar.textContent = 'Searching catalog and retrieving context…';

    // Prepare assistant response container
    const streamUI = render.appendAssistantMessage(messageThread);
    currentAbortController = new AbortController();

    let fullAnswer = '';

    await api.streamAsk(
      payload,
      {
        onMeta: (data) => {
          if (statusBar) {
            const modelInfo = data.model ? ` (${data.model})` : '';
            statusBar.textContent = `Streaming response${modelInfo}…`;
          }
          if (data.debug) {
            render.renderDebugDrawer(data.debug);
          }
        },
        onDebug: (debugData) => {
          render.renderDebugDrawer(debugData);
        },
        onToken: (token) => {
          fullAnswer += token;
          streamUI.updateToken(token);
        },
        onSources: (sources) => {
          streamUI.setSources(sources);
        },
        onDone: (summary) => {
          streamUI.finalize(fullAnswer);
          const latency = summary.latency_ms ? ` in ${(summary.latency_ms / 1000).toFixed(2)}s` : '';
          if (statusBar) statusBar.textContent = `Completed${latency}.`;
          setGeneratingState(false);
        },
        onError: (err) => {
          if (err.isAbort) {
            streamUI.finalize(fullAnswer ? fullAnswer + '\n\n*(Generation stopped)*' : '*(Generation stopped by user)*');
            if (statusBar) statusBar.textContent = 'Stopped by user.';
          } else {
            const msg = err.message || 'An error occurred during generation.';
            streamUI.finalize(fullAnswer ? fullAnswer + `\n\n> ⚠️ **Error:** ${msg}` : `> ⚠️ **Error:** ${msg}`);
            if (statusBar) statusBar.textContent = `Error: ${msg}`;
          }
          setGeneratingState(false);
        },
        onAbort: () => {
          streamUI.finalize(fullAnswer ? fullAnswer + '\n\n*(Generation stopped)*' : '*(Generation stopped by user)*');
          if (statusBar) statusBar.textContent = 'Cancelled.';
          setGeneratingState(false);
        },
      },
      currentAbortController.signal
    );
  }

  /**
   * Handle generation cancellation via Stop button.
   */
  function handleStop() {
    if (currentAbortController) {
      currentAbortController.abort();
      currentAbortController = null;
    }
    setGeneratingState(false);
  }

  /**
   * Reset conversation thread.
   */
  function handleNewChat() {
    if (isGenerating) {
      handleStop();
    }
    if (messageThread) {
      messageThread.innerHTML = `
        <div class="welcome-card" id="welcome-card">
          <h3>Welcome to Filumart Product Knowledge Assistant</h3>
          <p>Ask technical questions, compare models, verify specifications, or inspect supplier terms across packaging, refrigeration, and industrial equipment catalogs.</p>
          <div class="suggested-prompts">
            <button class="prompt-chip" data-query="What is the tape width capacity for PKG-120?">PKG-120 tape width capacity</button>
            <button class="prompt-chip" data-query="Compare packaging machine models and specifications in a table">Compare packaging machines</button>
            <button class="prompt-chip" data-query="Which chillers use R-410A refrigerant and what is their cooling capacity?">Chillers using R-410A</button>
          </div>
        </div>
      `;
      // Re-attach chip listeners
      document.querySelectorAll('.prompt-chip').forEach(chip => {
        chip.addEventListener('click', () => {
          const query = chip.getAttribute('data-query');
          if (query && queryInput) {
            queryInput.value = query;
            handleSend();
          }
        });
      });
    }
    if (statusBar) statusBar.textContent = '';
    queryInput?.focus();
  }

  /**
   * Set UI state between generating and idle.
   */
  function setGeneratingState(generating) {
    isGenerating = generating;
    if (btnSend) btnSend.disabled = generating;
    if (btnStop) btnStop.disabled = !generating;
    if (!generating) {
      currentAbortController = null;
    }
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
