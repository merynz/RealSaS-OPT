/* Existing proof-qualified native RSS delivery. No model or edited-runtime authority. */
import {apiFetch} from './api_gateway_v1.js';

export async function downloadQualifiedNativeRSS(bundle) {
  if (typeof bundle !== 'string' || !bundle.trim()) {
    throw new Error('A proof-qualified bundle is required');
  }
  const manifestResponse = await apiFetch('/api/export', {
    method: 'POST', cache: 'no-store',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({bundle}),
  });
  const manifest = await manifestResponse.json().catch(() => ({}));
  if (!manifestResponse.ok) {
    throw new Error(manifest?.message || manifest?.error || 'Native export unavailable');
  }
  if (
    manifest?.schema_version !== 'RealSaS.BustedQualifiedNativeDownload.v1' ||
    manifest?.authority !== 'EXISTING_PROOF_GATED_NATIVE_RUNTIME' ||
    manifest?.format !== 'rss' ||
    manifest?.edited_studio_state_included !== false ||
    !/^\/api\/blob\?/.test(String(manifest.download_url || '')) ||
    !/^[a-f0-9]{64}$/.test(String(manifest.content_sha256 || '')) ||
    !Number.isSafeInteger(manifest.size_bytes) || manifest.size_bytes <= 0
  ) {
    throw new Error('Unqualified native export receipt');
  }
  const source = await apiFetch(manifest.download_url, {cache: 'no-store'});
  if (!source.ok) throw new Error('Native RSS download failed');
  const raw = await source.arrayBuffer();
  if (raw.byteLength !== manifest.size_bytes) {
    throw new Error('Native RSS download size drift');
  }
  if (!globalThis.crypto?.subtle) {
    throw new Error('Secure SHA-256 verification is required for native export');
  }
  const digest = await globalThis.crypto.subtle.digest('SHA-256', raw);
  const actual = [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, '0')).join('');
  if (actual !== manifest.content_sha256) {
    throw new Error('Native RSS download SHA-256 mismatch');
  }
  const url = URL.createObjectURL(new Blob([raw], {type: 'application/octet-stream'}));
  try {
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'busted-character.rss';
    anchor.style.display = 'none';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  return {content_sha256: actual, size_bytes: raw.byteLength};
}
