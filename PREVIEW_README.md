# BUSTED / RealSaS actual interface inspection

This preview uses the **real Awaken and Studio frontend source** from PR #298 at commit 09d63cffde12325482b1bc9be9745523f083ca81. CSS and JavaScript (except the static-host safety guard) are exact Git object copies from that source ref.

**What it is:** a publicly loadable static browser presentation of the real frontend, including the original animated canvas effects, image upload preview, and Studio shell. No puppet bundles are synthesized.

**What it is not:** a compiler, a proof-passing product, a production Studio backend, or a fake success demo. Static hosting does not have a Database/Temporal/ModelRelease/Compiler endpoint. The otherwise present presentation-only fake PASS route in Awaken is disabled in this published preview. The production code on the source PR is not changed.

**Preview source:** https://github.com/merynz/RealSaS-OPT/pull/298

**To publish via GitHub Pages:** GitHub Settings > Pages > Deploy from a branch > `busted-live-preview` > /(root). Then open https://merynz.github.io/RealSaS-OPT/ (only after GitHub reports a successful deployment).

For non-production CDN inspection, the dev URL is https://raw.githack.com/merynz/RealSaS-OPT/busted-live-preview/index.html. The CDN provider may show an HTML safety confirmation.

This branch is an independent read-only preview source; changes to the production application remain in PR #298 and must pass the product's admission tests. Do not deploy this static shell as a real compile service.
