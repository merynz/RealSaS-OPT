# Initial forensic observations before executing the locked autopsy

These observations motivated the diagnostic but are not themselves its final result.

1. The materialized B-arm memory uses a fixed budget of 8192 tokens/subject: 384 pair + 384 ray-hit + 256 surface tokens per view (except one missing GreenBlob ray-hit token). Raw opportunity counts vary by orders of magnitude across views, so equal memory size does not imply equal evidence coverage.
2. Active authority is narrow: pair-front, global visibility, and source RGBA/boundary are populated; occlusion, deformation coherence, and attachment persistence remain abstentions. Therefore this B arm is a proxy for only part of the intended shared-scene relation hypothesis.
3. Manifest-level raw opportunity counts alone do not trivially explain the B effect. A full subject/view audit plus counterfactual resampling is required.
4. The original FIT8 corpus predates the current shared-scene/presentation hypothesis, so corpus adequacy is a live confound rather than a settled assumption.

No scientific conclusion should be drawn from this note without the autopsy result artifact.
