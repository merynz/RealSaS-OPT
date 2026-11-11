// The same main-owned Awaken/Studio UI works on the product server (same
// origin) and on a GitHub Pages/static preview (configured HTTPS API origin).
// A static GitHub Pages page NEVER implies that a compiler is available.
export function apiUrl(path) {
  if (typeof path !== 'string' || !path.startsWith('/api/') || path.startsWith('//')) {
    throw new TypeError('BUSTED_API_RELATIVE_PATH_REQUIRED');
  }
  const origin = String(globalThis.__BUSTED_PUBLIC_API_ORIGIN__ || '').trim();
  if (!origin) return path;
  let parsed;
  try { parsed = new URL(origin); }
  catch { throw new TypeError('BUSTED_API_ORIGIN_INVALID'); }
  const local = parsed.hostname === 'localhost' || parsed.hostname === '127.0.0.1';
  if ((parsed.protocol !== 'https:' && !(local && parsed.protocol === 'http:')) ||
      parsed.pathname !== '/' || parsed.search || parsed.hash ||
      parsed.username || parsed.password) {
    throw new TypeError('BUSTED_API_ORIGIN_NOT_ALLOWED');
  }
  return parsed.origin + path;
}

export function apiFetch(path, options = {}) {
  return fetch(apiUrl(path), options);
}
