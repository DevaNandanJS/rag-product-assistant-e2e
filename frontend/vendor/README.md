# Vendored Frontend Libraries

This directory contains pinned, vendor-reviewed client dependencies without external CDN reliance:

1. **marked.min.js**
   - Version: 15.0.12
   - Source: https://cdn.jsdelivr.net/npm/marked@15.0.12/marked.min.js
   - License: MIT License (Christopher Jeffrey)
   - Purpose: Progressive client-side markdown parsing for assistant answers.

2. **purify.min.js**
   - Version: 3.2.6
   - Source: https://cdn.jsdelivr.net/npm/dompurify@3.2.6/dist/purify.min.js
   - License: Apache-2.0 / MPL-2.0 (cure53)
   - Purpose: DOM sanitization of rendered markdown to prevent XSS. All citations and debug cards use native DOM construction (`createElement` / `textContent`).
