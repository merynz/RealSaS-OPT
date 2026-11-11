/* BUSTED public site -> single RSS.vNext product API.
 *
 * GitHub Pages is static. It never executes a compiler and may not claim
 * product readiness. Its only job is to forward /api/* requests to the
 * explicitly configured, HTTPS product service. A missing or invalid config
 * intentionally falls back to offline status.
 *
 * Every existing Awaken/Studio/Motion/Export consumer uses /api/*; routing
 * those paths here keeps the original UI and avoids a second frontend fork.
 */
(() => {
  'use strict';
  const nativeFetch = window.fetch.bind(window);
  const configURL = new URL('./site-config.json', document.baseURI).toString();
  const pagesOrigin = window.location.origin;
  const basePromise = nativeFetch(configURL, {cache: 'no-store', credentials: 'omit'})
    .then(async (response) => {
      if (!response.ok) return null;
      const settings = await response.json();
      if (settings?.schema !== 'RealSaS.BUSTED.SiteGatewayConfig.v1') return null;
      const text = settings?.api_base_url;
      if (text == null || text === '') return null;
      if (typeof text !== 'string') return null;
      const parsed = new URL(text);
      // No arbitrary URL schemes, credentials, path suffix or search args.
      // Loopback HTTP is allowed for local development ONLY.
      const local = ['localhost', '127.0.0.1', '[::1]'].includes(parsed.hostname);
      if (parsed.protocol !== 'https:' && !(parsed.protocol === 'http:' && local)) return null;
      if (parsed.username || parsed.password || parsed.search || parsed.hash
          || parsed.pathname !== '/') return null;
      return parsed.origin;
    })
    .catch(() => null);

  window.fetch = function bustedProductFetch(input, init = {}) {
    const source = typeof input === 'string' ? input : input instanceof URL ? input.href : input?.url;
    let url;
    try { url = new URL(source, document.baseURI); }
    catch { return nativeFetch(input, init); }
    if (url.origin !== pagesOrigin || !url.pathname.startsWith('/api/')) {
      return nativeFetch(input, init);
    }
    return basePromise.then((origin) => {
      if (!origin) return nativeFetch(input, init); // Explicit offline fallback.
      const replacement = origin + url.pathname + url.search;
      if (typeof input === 'string' || input instanceof URL) {
        return nativeFetch(replacement, init);
      }
      // Retain Request method/body and AbortSignal when translating origin.
      return nativeFetch(new Request(replacement, input), init);
    });
  };
})();
