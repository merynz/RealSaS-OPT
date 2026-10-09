# Slot-owned attachment depth

The active scoped research graph uses `presentation_research_v3`. It preserves
V4 canonical pose blending and the witness-wide bounded body SE(2) repair, then
compiles V5 target attachment depth. Mechanics, rig, skin and retargeting are
unchanged. The unarmed body preset does not supply target prop placement; the
target rest artwork and its qualified canonical slot own that placement.

For each view, bind canonical rest depth `z0(v)` once from the safe M anchors.
For a target rigid attachment with canonical slot `s`, evaluate:

```
XY(v,t) = R2(s,t) XY0(v) + translation2(s,t)
Z(v,t)  = z0(v) + cameraDepth(posedSlot(s,t)) - cameraDepth(restSlot(s))
```

All disconnected source-art charts of that attachment share the same slot
transform. Rest depth relief is invariant. No animated M-surface depth remains
on the attachment. This is a camera-facing 2.5D artwork contract: out-of-plane
foreshortening is not inferred from the unarmed source animation.

Overlap remains canonical camera depth ascending, with nearest nontransparent
fragment ownership and front-to-back premultiplied compositing. No face-order
or object-order bias is introduced. Ties at 1e-9, layer overflow above 32 and
nonpositive depth still fail closed, including internal attachment overlaps.

Stage45 V3 rebinds safe anchors from Stage37, rederives vertex/face owners and
rest depth, replays each slot XY/depth transform, checks unchanged body depth,
and independently replays the canonical palette and body repair. The ownership
verdict is measured over the full clip/view/frame matrix and participates in
the final gate. V1/V2 historical proof flags do not certify this new contract.
The V5 native consumer requires typed attachment ownership and depth contracts
and rejects missing, malformed or out-of-range face-owner entries.

The Attempt4 native/reference failure was a single green-byte difference at
pixel (x=113,y=58), repeated at IDLE/V4 frames 0 and 80. Coverage and face owner
were identical. Straight RGB now uses the native consumer's shared reciprocal
alpha arithmetic in the independent reference; strict byte parity is retained.
The adversarial half-byte fixture verifies this without a Knight-specific patch.

Validation remains scoped to the sealed Knight FIT1 motion witness. Absolute G3
and product/generalization authority remain open. Fresh downstream Go Attempt
identities and the final scientific verdict are recorded after execution.
