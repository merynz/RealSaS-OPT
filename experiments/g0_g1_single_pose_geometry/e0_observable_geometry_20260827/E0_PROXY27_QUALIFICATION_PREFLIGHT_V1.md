# E0 Proxy27 Qualification Preflight V1

Status: **PASS — Proxy27 downstream outcomes unopened.**

- notebook SHA-256: `583790ccb53858c11b48ab4f3f433f5e6a01495b9cadeac9d7f79c16e4815975`
- qualification runner SHA-256: `505420160174ad236355ab62106273dde90a72bbe3d4eac498e4292d9df80d58`
- frozen intersection SHA-256: `fcbdd90d585a3845b8f799fbd1ba6dff526d4714aa581a7f6d301da86d2ecbd9`
- non-binding expectation SHA-256: `5c9cbecad2496c8145515f31a04f0bd8dbce85a0078e47e25d1acf8b04725b04`
- diagnostic-result SHA-256: `906e6e686aa2e82e3e27851b4d5f10dc5566b8c55b583a73f0b6e379dcf4370d`
- V1.3 checkpoint seal SHA-256: `5d9dcde8e68178b98e8fae3002912e8c895dcc495578278dbf57c577397e8856`
- six checkpoint bytes: **6/6 SHA PASS**
- notebook `PREFLIGHT_ONLY=1`: **PASS**
- exact evaluator calibration replay on `asset_f8a40d6c5d815fe79c8b5e42`, Arachne + Geppetto × D0/D1/D2: max absolute numeric drift **1.0431e-7**
- scientific training executed: **false**
- 437 official pack rebuild executed: **false**
- Proxy27 downstream metric evaluation executed: **false**
- DEV32 opened: **false**

The next executable action is the exact notebook above on Colab GPU. Its runner builds only the 27 qualification packs, fail-closes on symmetric target unavailability before metrics, then opens the frozen Proxy27 evaluation exactly once.
