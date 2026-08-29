# RealSaS DINO Controlled Ladder — Zero-Step Closure

Date: 2026-08-29

## Decision

**ZERO-STEP PREFLIGHT: PASS**

**CONTROLLED DINO S/B/L/g TRAINING: AUTHORIZED**

No scientific optimizer step was taken before this authorization. DEV32 remains closed.

## Closed gates

- TRAIN512 native-1024 authority: 512/512
- official DINOv2 S/B/L/g raw weights + exact source + preprocess authority: PASS
- shared learner/init/sample/cache apparatus freeze: PASS
- DTB-ND1 carrier semantic parity: 6/6 PASS
- real DINO token online repeat/cache/Q_d+L2 parity: S/B/L/g all PASS

## Locked apparatus

- shared trainable parameters: 5,046,739
- target: camera-forward depth d only
- reconstruction: P = O + dF
- TRAIN512 set SHA: 1958fa5ed80430ac8ae8f9e66f8d94bc5bfe89c74b5553d13b891c87fdefb2a2
- deterministic sample stream data SHA: 2aca93a8840199785cb4392f52e5ef449b26d31cb52a10473e254aa59a4517bd
- matched checkpoint: step 7168
- primary checkpoint: step 32768
- primary consumer metric: frozen DTB-ND1 direct replay
- DEV32: CLOSED

## Canonical hashes

Combined seal content SHA-256: `73d6bbb8d4656c6f71ed6edbd3fbd4802361041fe78dd1eaa2f8e0b712bf95a9`
Combined seal raw SHA-256: `9aa448a48e4168b1c161962093bf6ee36f527a27f1ded13ac445deca97d3b8fa`

Training authorization content SHA-256: `87bc72631623caed1f563439054f937db3011a90b4f2b2be79cfbc4034a5f086`
Training authorization raw SHA-256: `9d6b4533fcb10c6cdd2ea0f3a8aa08577a1feca74695a3764517d5bfeb6fffb6`

## Next

Execute the controlled ladder in sealed order:

`S -> B -> L -> g`

No architecture, preprocessing, membership, optimizer/loss, sample-stream, cache-semantic, or candidate-specific tuning changes are permitted after the token-output opening.
