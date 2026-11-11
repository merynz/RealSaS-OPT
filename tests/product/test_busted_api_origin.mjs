import assert from 'node:assert/strict';
import {apiUrl} from '../../product/busted_site/static/js/api_gateway_v1.js';

globalThis.__BUSTED_PUBLIC_API_ORIGIN__ = '';
assert.equal(apiUrl('/api/project'), '/api/project');
assert.equal(apiUrl('/api/runtime/frame?bundle=token&time=0.5'),
             '/api/runtime/frame?bundle=token&time=0.5');
globalThis.__BUSTED_PUBLIC_API_ORIGIN__ = 'https://api.example.org';
assert.equal(apiUrl('/api/project'), 'https://api.example.org/api/project');
globalThis.__BUSTED_PUBLIC_API_ORIGIN__ = 'http://localhost:8765';
assert.equal(apiUrl('/api/project'), 'http://localhost:8765/api/project');
for (const forbidden of [
  'http://api.example.org', 'https://api.example.org/private',
  'javascript:alert(1)', 'https://evil@example.org', 'https://api.example.org/?token=secret',
]) {
  globalThis.__BUSTED_PUBLIC_API_ORIGIN__ = forbidden;
  assert.throws(() => apiUrl('/api/project'), /ORIGIN_/);
}
globalThis.__BUSTED_PUBLIC_API_ORIGIN__ = '';
for (const forbidden of ['//evil.example/api/project', 'https://evil.example/api/project', '/studio.html']) {
  assert.throws(() => apiUrl(forbidden), /BUSTED_API_RELATIVE_PATH_REQUIRED/);
}
console.log('BUSTED_PUBLIC_API_ORIGIN_POLICY_PASS');
