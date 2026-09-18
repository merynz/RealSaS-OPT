# Mage observation-framing contradiction — 2026-09-18

**Status:** `FAIL__CURRENT_MAGE_FULL_SUBJECT_OBSERVATION_AUTHORITY_IS_BODY_ONLY_FRAMED`

The current eight-view Mage observation authority is not a complete full-subject observation set. Exact forensic replay shows that the camera was inherited from the historical 3348-vertex BODY-only authority while the rendered scene already contained the broader Mage subject.

The current camera center is `[0, 0.1144906580, 1.1013257504]`. The historical BODY-only bbox center is the same to numerical precision, and the current `half_extent=1.2335147476` is exactly ~`1.12x` the BODY-only max-axis half span `1.1013524005`.

This framing error is visible directly in the admitted product observations: all eight alpha masks touch the top border; six of eight also touch a left/right border. The common bottom row is 969.

The corrected full supported FIT teacher has 5279 vertices / 5683 faces. Under the current BODY-derived center/half, 119 teacher vertices lie outside the H1 `[-1,+1]^3` normalized field domain: BODY 0, BOOK 29, WAND/STAFF 38, HAT 52, CAPE 0. Therefore part of the full subject was not merely difficult for H1 to learn; it was outside the representable field domain.

This finding reopens the current Mage full-subject product authority from the observation/camera apparatus forward. Existing H1/GSA/Geppetto/Arachne artifacts are preserved as historical scoped evidence and are not rewritten. They may not authorize full-subject Mage product proof under the cropped observation authority.

No threshold relaxation is authorized. The next required evidence is a same-style, same-subject, same-yaw 8-view rerender with camera framing derived from the complete admitted render subject, followed by a new observation/camera audit before any H1/GSA/S/G/W reclosure.
