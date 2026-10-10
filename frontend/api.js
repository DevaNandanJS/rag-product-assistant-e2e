/**
 * Filumart Assistant - API Client
 * Zero-framework SSE streaming reader and backend REST endpoints.
 */

/**
 * Fetch catalog filters from GET /filters.
 * @returns {Promise<{categories: string[], suppliers: string[], countries: string[], products: string[]}>}
 */
export async function fetchFilters() {
  const resp = await fetch('/filters');
  if (!resp.ok) {
    throw new Error(`Failed to load filters: ${resp.status} ${resp.statusText}`);
  }
  return await resp.json();
}

/**
 * Fetch system health status from GET /health.
 * @returns {Promise<{status: string, llm_mode: string, version?: string}>}
 */
export async function fetchHealth() {
  const resp = await fetch('/health');
  if (!resp.ok) {
    throw new Error(`Failed to fetch health: ${resp.status} ${resp.statusText}`);
  }
  return await resp.json();
}

/**
 * Stream an assistant query via POST /ask/stream with SSE.
 * 
 * @param {Object} payload Query request body ({ query, filters, debug, top_k })
 * @param {Object} callbacks Event handler callbacks
 * @param {Function} [callbacks.onMeta] Called on 'meta' event with { request_id, model, ... }
 * @param {Function} [callbacks.onDebug] Called on 'debug' event with retrieval debug info
 * @param {Function} [callbacks.onToken] Called on 'token' event with token string
 * @param {Function} [callbacks.onSources] Called on 'sources' event with source citation list
 * @param {Function} [callbacks.onGrounding] Called on 'grounding' event with grounding info
 * @param {Function} [callbacks.onDone] Called on 'done' event with completion summary
 * @param {Function} [callbacks.onError] Called on error with error object/message
 * @param {AbortSignal} [signal] Optional AbortSignal from AbortController
 */
export async function streamAsk(payload, callbacks = {}, signal = null) {
  try {
    const resp = await fetch('/ask/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify(payload),
      signal,
    });

    if (!resp.ok) {
      const errText = await resp.text();
      let errMsg = `HTTP ${resp.status}: ${resp.statusText}`;
      try {
        const errJson = JSON.parse(errText);
        if (errJson.detail) errMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
        else if (errJson.message) errMsg = errJson.message;
      } catch (_) {
        if (errText) errMsg = errText;
      }
      callbacks.onError?.({ message: errMsg, status: resp.status });
      return;
    }

    if (!resp.body) {
      callbacks.onError?.({ message: 'Response body is not readable' });
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      
      // SSE frames are separated by double newlines (\n\n or \r\n\r\n)
      const frames = buffer.split(/\r?\n\r?\n/);
      // The last element is an incomplete frame, retain in buffer
      buffer = frames.pop() || '';

      for (const frame of frames) {
        if (!frame.trim()) continue;
        parseAndDispatchFrame(frame, callbacks);
      }
    }

    // Flush any remaining buffer if complete
    if (buffer.trim()) {
      parseAndDispatchFrame(buffer, callbacks);
    }

  } catch (err) {
    if (err.name === 'AbortError') {
      callbacks.onAbort?.();
      callbacks.onError?.({ message: 'Generation stopped by user.', isAbort: true });
    } else {
      callbacks.onError?.({ message: err.message || 'Unknown network error', error: err });
    }
  }
}

/**
 * Parse an SSE frame string and dispatch to relevant callback.
 */
function parseAndDispatchFrame(frame, callbacks) {
  const lines = frame.split(/\r?\n/);
  let eventType = 'message';
  const dataLines = [];

  for (const line of lines) {
    if (line.startsWith(':')) {
      // SSE comment / ping - ignore
      continue;
    }
    if (line.startsWith('event:')) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (dataLines.length === 0) return;

  const rawData = dataLines.join('\n');
  let data;
  try {
    data = JSON.parse(rawData);
  } catch (e) {
    data = rawData;
  }

  switch (eventType) {
    case 'meta':
      callbacks.onMeta?.(data);
      break;
    case 'debug':
      callbacks.onDebug?.(data);
      break;
    case 'token': {
      const tokenText = typeof data === 'object' && data !== null
        ? (data.content !== undefined ? data.content : (data.token !== undefined ? data.token : ''))
        : String(data ?? '');
      callbacks.onToken?.(tokenText);
      break;
    }
    case 'sources':
      callbacks.onSources?.(data.sources || (Array.isArray(data) ? data : []));
      break;
    case 'grounding':
      callbacks.onGrounding?.(data);
      break;
    case 'done':
      callbacks.onDone?.(data);
      break;
    case 'error':
      callbacks.onError?.(data);
      break;
    default:
      callbacks.onMessage?.({ event: eventType, data });
      break;
  }
}

// Global attachment for fallback scripts
if (typeof window !== 'undefined') {
  window.FiluAPI = {
    fetchFilters,
    fetchHealth,
    streamAsk,
  };
}
