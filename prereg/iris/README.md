# IRIS Preregistration Status

During `audit/iris-architecture-discipline-20260824`, existing G1/Controlled-V1 preregistrations are historical evidence only. They do not authorize V2 training.

No `IRIS_SINGLE_POSE_V2_MINI_PREREG` exists yet by design.

A V2 mini prereg may be created only after:

- V2 committed-byte preflight PASS;
- real-corpus no-optimizer census PASS;
- GPU 256/512/1024 forward-capacity preflight PASS/explicit bounded decision;
- mini membership frozen;
- evaluator metric units and checkpoint selection key frozen.
