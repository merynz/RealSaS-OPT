# RealSaS — IRIS Reprojection V2 Real Gate0 Membership Hash Verifier Erratum — 2026-08-31

**Status:** `VERIFIER_BUG_CONFIRMED__SCIENTIFIC_POPULATION_UNCHANGED__HASH_SCHEME_REINTERPRETATION_FORBIDDEN`

## Trigger

The FAST Gate0 notebook located the exact sealed DINO membership artifact and passed:

- raw file SHA-256 `ae1edcaad47cf4867f11d51305eb57db06acb63ef13fc60ca3263c043ef8f71c`;
- stored/canonical content SHA-256 `305bcc2efc863c035ad7bce6e0afb29f2354ecc57abd1746e7b15b0485f71d24`;
- exact `train_order_512` count `512` with no duplicates.

It then failed a newly introduced check that recomputed a locally defined `sorted(ids) + newline` digest and compared it to the historical declared TRAIN512 set SHA `1958fa5ed80430ac8ae8f9e66f8d94bc5bfe89c74b5553d13b891c87fdefb2a2`.

## Root cause

The historical DINO membership authority already carries the field:

`train512_set_sha256 = 1958fa5ed80430ac8ae8f9e66f8d94bc5bfe89c74b5553d13b891c87fdefb2a2`.

The sealed DINO ladder validator deliberately treated that field as historical authority and explicitly did **not** redefine its construction through the later local `set_sha` helper.

The Real Gate0 runner/notebook incorrectly reused a convenient local sorted-newline digest implementation as though it were the historical set-hash algorithm. That is a verifier-semantics bug, not evidence of membership drift.

## Binding correction

Scientific Gate0 membership validation is:

1. exact raw membership file SHA matches the frozen authority;
2. `content_sha256` field matches the frozen authority;
3. canonical content hash recomputed with `content_sha256` removed matches the same frozen authority;
4. `train_order_512` has exactly 512 unique IDs;
5. the exact membership artifact's declared `train512_set_sha256` field matches the frozen historical value;
6. after execution, measured asset IDs must equal the authoritative TRAIN512 IDs as an exact set with no duplicates.

A newly computed sorted-newline SHA may be emitted only as a **diagnostic with an explicit scheme label**. It may not be compared to the historical declared set SHA unless the historical construction algorithm is independently recovered and frozen.

## Scientific scope

This erratum does **not** change:

- the TRAIN512 population;
- any asset membership;
- the membership raw/content hashes;
- Gate0 cameras, masks, raster truth, truth-row sampling, hull paddings, volume sampling, spacing candidates or thresholds;
- optimizer/training authorization.

No scientific Gate0 result had been produced before this correction. The failed notebook stopped before the real-corpus measurement.

## Implementation rule

The original `gate0_real_corpus_v1.py` remains historical sealed source. Scientific execution should use the hash-authority corrected wrapper `gate0_real_corpus_v1_1_hashfix.py` or an exactly equivalent notebook implementation bound to this erratum.
