# BUSTED · Live Awaken / Studio design preview

This GitHub Pages branch deploys the actual frontend shell from the RealSaS productization work, with a **preview-only visual layer** under active development.

- **Awaken:** sparse source-intake UI, front/back assignment and local image preview; a CSS perspective grid and far-field terminal text typed in place.
- **Studio:** the real editor shell with BUSTED navigation. No sample puppet, compile, proof or export output is fabricated.
- **Backend:** GitHub Pages is static and has no production compiler or proof gateway. Awaken explicitly reports the offline gateway and keeps source compile disabled.
- **Performance:** the older random particles, radial streaks, text wobble and 24 Hz full-screen Canvas drawing were removed. Fixed CSS depth and an event-driven typewriter replace them. The page stops the typewriter while hidden and honors reduced-motion preferences.
- **Authority:** the standalone Studio route is browsable even without a bundle, but a genuine compiled puppet still requires ProductProof and a qualified workspace handoff.

Site: https://merynz.github.io/RealSaS-OPT/

Product code change: https://github.com/merynz/RealSaS-OPT/pull/332 (draft).
Parent fail-closed safety PR: https://github.com/merynz/RealSaS-OPT/pull/331 (draft).
Productization branch: https://github.com/merynz/RealSaS-OPT/pull/298 (draft).

Preview changes intentionally do not modify the RealSaS compilation authority. The frontend is **not** an inference service, model checkpoint, or production deployment.
