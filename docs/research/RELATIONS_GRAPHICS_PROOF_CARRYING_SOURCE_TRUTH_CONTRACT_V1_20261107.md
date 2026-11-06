# RealSaS — Proof-Carrying Relations × Graphics Source Truth Contract V1

**Research scope:** finite original-source evidence / specified intervention and representation grammars. **Authority:** source-only; original artist drawable owner UNKNOWN; IRIS/TESSA training and materialization NOT authorized.

This is the first explicit mathematical contract for separating what the original art establishes from what a compiler chooses. The statements below are proved *under declared observation and grammar assumptions*, not a proof that arbitrary source art has one unique 2D layer partition or that a one-image network can recover hidden truth.

## I. The object we are actually trying to infer

Let an original source world be \(w\), containing geometry, appearance, material/UV, articulation, cameras and source interventions. Let \(e=(s,\ell,u,p,v,h,\rho)\) be a measurement address:

- \(s\): immutable original-source SHA and provenance chain;
- \(\ell\): **typed locus** — indexed source face/edge, vertex-pair, face-corner UV, shader palette address, original Cycles sample-pixel, or aggregate measurement. Never equate their IDs without an explicit address map;
- \(u\): source operation (none, source transform, original UV RGB edit, original palette edit, hide/reveal, etc.);
- \(p,v,h\): original pose, camera and precise sampling/ray policy. V18j 4×4 rays are not automatically original Cycles samples; Cat bind/rest and authored pose14 are not the same condition;
- \(\rho\): exact independent original source court SHA, source pixel-origin and shader replay policy.

An observation is the partial map

\[
E_s(e)=(y_e,\ m_e),\qquad
m_e\in\{\mathrm{QUALIFIED},\mathrm{UNKNOWN},\mathrm{UNOBSERVABLE},\mathrm{VIOLATED}\}.
\]

The UNKNOWN value is **NOT Boolean false**. A source edit whose real pixel consequences were hidden by occlusion is not a negative edit or a same-drawable relation. A mistaken source-UV V origin is a failed instrument until separately requalified.

The canonical **source Relations/Graphics** family is a typed, individually masked vector:

\[
R(\ell_i,\ell_j\mid u,p,v,h)
=
(\underbrace{T,W,N,M}_{\text{topology, source-world proximity, normals, motion}},
 \underbrace{U,A,Z}_{\text{UV, source appearance, view-local depth/visibility}},
 \underbrace{I,O}_{\text{original edit consequence, observability}})
\]

where each component carries its own observation scope and proof receipt. **Artist-authored drawable ID is not a field.** A later compiler can choose a packet grouping using a capability contract; it is not a source observation.

### Identifiability criterion

Let \(\mathcal W(E,A)\) be all concrete worlds satisfying qualified evidence \(E\) and declared assumptions \(A\). A source query \(Q\) is **identified** if and only if

\[
\forall w_1,w_2\in\mathcal W(E,A):\quad Q(w_1)=Q(w_2).
\]

When two admissible worlds have the same observations but disagree on a query, its correct truth value is **NOT IDENTIFIED**, rather than a guessed binary label. This follows the general identifiability concept in [Pearl, causal inference](https://escholarship.org/uc/item/18j2x5zr). Conservatively keeping unknown properties is also aligned with [Cousot and Cousot, abstract interpretation, 1977](https://doi.org/10.1145/512950.512973). The actual source-locus contract and results below are RealSaS-specific proposals, not published proofs from those authors.

**Three disjoint authorities:**
1. **E — source truth:** independently measured original geometry, ray/appearance facts, control and edit consequences; typed provenance and valid masks.
2. **C/G — capability and executable grammar:** required operations and what a grouped, masked, layered or geometry-depth carrier can perform.
3. **P — representation policy:** runtime cost, simplicity, graphics packet count, edit ergonomics and stability among feasible alternatives.

The only admissible path is

\[
E\ \to\ \operatorname{IdentifiedConstraints}(E,C,G)\
\to\ \operatorname{ChooseRepresentation}(E,C,G,P).
\]

The final representation decision does not automatically produce a source artist truth label or IRIS one-view inference target.

## II. Six mathematical claims with proofs and exact boundaries

### Theorem 1 — Exact unique coarsest quotient for a COMPLETE uniform Boolean workload

For \(n\) original loci and \(m\) independently admitted Boolean source edit tasks, let \(B\in\{0,1\}^{n\times m}\). A **uniform-handle grammar** requires all source loci inside one handle to share one selected/unselected value for every task.

Define \(i\sim j\) exactly when \(B_{i,:}=B_{j,:}\).

**Claim:** \(\pi^\star = L/{\sim}\) is the unique coarsest feasible uniform-handle partition. Every other feasible partition refines \(\pi^\star\). Its minimum packet count is the number of different complete row signatures.

**Proof:** If \(i,j\) share a feasible handle, their responses must match under each of the \(m\) operations, so they lie in one equivalence class of \(\sim\). Conversely, each complete signature class responds uniformly to all \(m\) operations and is therefore feasible. Every feasible grouping refines the signature quotient and the quotient is itself feasible. QED.

**Executable proof certificate:** For every two different complete signatures emit the exact source operation and the two differing Boolean values; the algorithm uses no artist semantic names. The complete-quotient kernel is linear in locus/task count plus pairwise DISTINCT profile certificate generation. It does **not** require enumerating the Bell-number of all representations.

**Actual original-source qualification:** PR #351's independently measured Cat native control workload has 7 coarsest atoms for 24 axis tasks; Triceratops 245 for 117 native tasks. The additional original Triceratops source constant-color palette family raises its declared workload to 257 atoms for 120 tasks. The Triceratops report includes 70 UNKNOWN face placeholders and 85 unknown FIT face-task cells: those placeholders are not a globally proved complete signature quotient. The atom numbers are **relative to that exact policy and instrumentation**; they are not intrinsic author layer counts.

### Theorem 2 — Partial evidence is a conflict graph, NOT an equivalence partition

For \(B_{it}\in\{0,1,?\}\), define a witnessed conflict edge \(ij\in H\) iff an independently qualified task \(t\) has known opposite values for both loci.

**Claim:** A grouping is feasible for the observed task data in the uniform grammar iff no conflict edge lies inside any group. Therefore the minimum number of uniform packets **on the observed finite workload** equals the chromatic number \(\chi(H)\).

**Proof:** Putting an observed 0 and 1 in one uniform handle violates the corresponding operation. Conversely, an edge-free group never has an observed 0/1 contradiction in any task; choose its one observed value, if present, and otherwise any Boolean value. This satisfies every observed task. Thus feasible partitions are exactly proper colorings. QED.

Compatibility is not transitive. For \(A=(1,?), B=(?,0), C=(0,0)\), pairs (A,B) and (B,C) are compatible, but (A,C) conflict. Thus **no-conflict is NOT a source must-link.** Our bounded exhaustive enumerator checks every set partition for \(n\le8\), yields all optimal partitions and exact counterexample edit tasks. It ABSTAINS from claiming a chromatic minimum for larger incomplete graphs until a separately verified exact graph solver exists.

### Theorem 3 — Masked and split carriers make packet ownership unidentifiable

Consider two representations that preserve the **same original per-face geometry, UV/appearance, depth-sorted compositing and independent edit-address masks**, but assign different packet labels:

- \(G_{\mathrm{masked}}\): one carrier packet, multiple independent face/texel masks;
- \(G_{\mathrm{split}}\): two or more packets, same per-face data and masks, same sample-local ordering.

If the actual executor uses the identical per-face values and depth/alpha compositing regardless of these packet labels, then for every operation, pose, view and sample in its domain,

\[
F_{\mathrm{masked}}(u,p,v,h)=F_{\mathrm{split}}(u,p,v,h)
\]

although packet assignments differ.

**Proof:** Every per-face input and original edit mask is equal, and packet labels are not inputs to the compositing calculation. Therefore corresponding source contributions and every final pixel are equal for all allowed operations. Different packet maps have the same observable behavior. QED, by explicit two-model construction.

This is a **conditional constructive nonidentifiability theorem**: if a capability instead requires each packet to transform independently, be drawn by an order-limited sprite-only renderer, or carry release/attachment states, additional operations can distinguish candidates. Those are *capability-specific requirements*, never proof of historical artist drawing intent.

Concrete original tests already demonstrate why this distinction matters. Cat has 3 independently required source-UV edit controls under PR #354's eight-state contract, but not 3 physically necessary drawables. Triceratops has 3 independent original shader color operations, yet their source material membership can be spatially disconnected.

### Theorem 4 — A common source SHA is insufficient for an operational causal join

For two source evidence facts to jointly determine a **local** source consequence, they must share or have a **separately proved mapping between** source locus addresses, operation, pose, camera, sampling/ray protocol and renderer policy.

**Counterexample proof:** The V18s Cat relation graph is indexed by original 684-face and UV/vertex pairs in FIT authored poses 0/1. The source Cat UV physical-edit ledger stores *aggregate source sample-pixel incidents*, across REST camera V1/V3/V5/V7 and same-Cycles sample offsets. Consider two hypothetical local source-face causal assignments that exchange which of two original faces affects a measured edit pixel while retaining the SAME total per-view changed-pixel counts and per-edit overlap aggregates. They agree on all existing aggregated observations but assign different local face edit effects. Therefore the face-local causal relation is not identified by these aggregate receipts. QED.

The source SHA proves they are about one original Cat; it cannot fabricate the missing **sample-pixel→original face/corner** causal correspondence. The executable scope gate emits **SUBJECT_LEVEL_ONLY / NO_LOCAL_CONSEQUENCE_JOIN**, not a false same-face/source-owner label. Conversely, different original SHA files may not be compared as if they have shared source face IDs.

### Theorem 5 — Qualified evidence is monotonically restrictive

Let \(E'\) consistently extend \(E\) with independently valid source Boolean responses. The witnessed conflict graph can only add edges: \(H(E)\subseteq H(E')\). Its proper colorings can only be removed, and \(\chi(H(E'))\ge\chi(H(E))\).

**Proof:** Previously observed contradictory 0/1 source pairs remain contradictory under an extension that preserves the original observations. A partition violating an earlier conflict cannot become valid after adding facts. Thus the feasible set shrinks and the minimum uniform packet count cannot decrease. QED.

**Boundary:** If an old source measurement was incorrect (e.g. first Cat glTF UV V-to-PNG source convention), an amendment replaces an invalid experimental premise and records a NEW court result. That is not a consistent evidence extension and not a violation of this theorem. Original FAIL must be retained.

### Theorem 6 — A hidden source relation need not be identifiable from one rendered image

Suppose two original source worlds have the same observed 2D input image \(x\) but differ in hidden geometry, occluded source material state, intervention response or original author metadata. For any deterministic model \(f\), \(f(x)\) is the same under both worlds, so it cannot match two unequal target values.

**Proof:** \(x_1=x_2\Rightarrow f(x_1)=f(x_2)\), while target \(Q(w_1)\neq Q(w_2)\). At least one prediction must be incorrect. QED.

This does **not** rule out probabilistic IRIS learning, good priors, useful representation prediction or confidence-calibrated targets; it rules out asserting that teacher-private unobserved relations are **uniquely implied by a single image** without additional assumptions.

### Theorem 7 — A COMPLETE finite source-intervention truth has a UNIQUE interaction decomposition (Möbius inversion)

This is a **positive identification result** beyond the limitations of drawable grouping.

Let \(U=\{u_1,\dots,u_m\}\) be \(m\) independently implementable original-source Boolean edits. Suppose all \(2^m\) originally physically rendered intervention combinations are observed at the EXACT SAME source, pose, camera, renderer policy and sample index. Fix any qualified pixel/ray and let \(Y(S)\in V\), \(S\subseteq U\), be the source linear-premultiplied RGBA vector for edited subset \(S\). Here \(V=\mathbb R^4\), or the entire aligned observed source image vector space.

Define for each intervention subset \(A\):

\[
\boxed{\quad
\Delta(A)=\sum_{B\subseteq A}(-1)^{|A|-|B|}Y(B).
\quad}
\]

Then every independently observed source edit render has the exact reconstruction

\[
\boxed{\quad Y(S)=\sum_{A\subseteq S}\Delta(A).\quad}
\]

**Proof:** Substitute the first equation into the second and exchange the two finite sums. The coefficient on any \(Y(B)\) becomes
\(\sum_{A:\,B\subseteq A\subseteq S}(-1)^{|A|-|B|}=(1-1)^{|S|-|B|}\), which is 1 when \(B=S\) and 0 otherwise. Thus the output is exactly \(Y(S)\). This is an invertible upper-triangular transform with unit diagonal, so **the interaction coefficients are unique** in this complete finite intervention and measurement domain. QED.

The terms have clear *operational* meaning: \(\Delta(\emptyset)\) baseline, \(\Delta(\{i\})\) independent first-order source edit effect, \(\Delta(\{i,j\})\) nonadditive pair interaction, and \(\Delta(\{i,j,k\})\) remaining third-order interaction. There is NO source author drawable or layer identity in this algebra. A 3-control source can have eight uniquely observable operation states without requiring any particular layer count.

**Unknown and applicability:** If a required subset \(Y(B)\) is missing, the Möbius coefficients depending on it are not identified from that incomplete table without additional assumptions. At larger \(m\) the full factorial requires \(2^m\) source renders; choose a bounded preregistered operation scope or adaptive valid experiment design. Nonlinear alpha/depth compositing is allowed inside measured \(Y(S)\) as long as all observations are in the same *output vector space* and have the same source sample alignment. Nonlinear scene interactions then appear as nonzero higher-order \(\Delta\), not as an assumption of additivity.

#### Real original-source Cat full factorial (rest pose, NOT IRIS student)

Original #351 physically rendered all 8 combinations of THREE disjoint 64×64 source atlas RGB-invert edits in four original cameras and eight exactly matched Cycles sample offsets: **256 original physical rendered arms**; array SHA d422b7d7d082fba5deac5dd1239b8c6aec078f4d635a7bd5b1f10e514100b114, independent original court SHA 90b85b9c1a963b23f7c3037ca443975bf2d5c5d4123568d8aa8e54557e433c08.

Direct source-array measurement, now independently validated on unchanged original source bytes by [GitHub Actions run 42298052382](https://github.com/merynz/RealSaS-OPT/actions/runs/42298052382), yields the following exact measured finite-domain result:

- First-order effects \(\Delta(\{0\}),\Delta(\{1\}),\Delta(\{2\})\) exceed the precommitted 0.025 premultiplied RGBA threshold at **290,979 / 264,554 / 6,048** original sample-pixel incidences; 0 and 1 share **2,799** projected source sample incidences.
- Pair interaction \(\Delta(\{0,1\})\) has nonzero maximum channel magnitude about **0.005160898**, but **0** source sample-pixel incidents exceed 0.025.
- Pair effects \(\Delta(\{0,2\})\), \(\Delta(\{1,2\})\) and third-order \(\Delta(\{0,1,2\})\) are exactly zero in the observed stored source sample arrays under the measured PM conversion; **0** threshold incidents.
- Full Möbius inversion reconstructs every stored physical source arm to floating-point precision; this is an algebraic identity, NOT an independently heldout new-render generalization test.
- The source 0/1 projected supports do overlap, yet their measured nonadditivity is small. Hence **projected effect overlap does not imply nonlinear interaction or a necessary drawable separation**.

**Independent original physical source proof PASS:** producer source output SHA bf4256bb596a56cce731a3ae97d9294c9c92876a73e7dacf19aad175ba7d35cb; separate inclusion–exclusion court SHA d6ab439c66824d2798bc6aaaa5e584d68c4b4c3277abd40ccbd4e6dbe9588185. Two workflow jobs successful; original bytes private, SHA readback verified. This court validates the bounded original Cat source edit-interaction result, not arbitrary pose/general artist grouping.

**Physical verification lane:** producer \`tools/corpus/source_intervention_mobius_truth_v1.py\`; independent closed-form inclusion-exclusion court \`tools/corpus/court_source_intervention_mobius_v1.py\`; new GitHub source workflow \`.github/workflows/source-intervention-mobius-original-cat-v1.yml\`. These use SHA-immutable original source samples, validate the original independent court, perform two distinct mathematical computations, preserve original 0.025 tolerance and avoid any source face/artist inference.

This theorem gives a mathematically identified and source-physically measurable **edit-consequence relation algebra**, complementary to the earlier mechanical/source-UV and view-depth relations. It is a concrete component of the generic Relations/Graphics truth contract, not a synonym for puppet GraphicsIR segmentation.

## III. Exact physical-source reality check

The source court uses the following previously independently sealed ORIGINAL documents, with a separate SHA for each producer and independent verdict:

- Cat V18s typed source-world/UV/response report SHA cbf44fee069e79ea3a7591669fede47befe0fb0ff7eab02aa4d5b3479a75a829;
- independent V18s court SHA c35ab620e3cf0955bb0b8ca445838d0320b50240bf357a706385eeeb1b67e09c;
- Cat physical original linked-texture edit ledger SHA 60ce8085c91282d43e2a16147d08dd6883764a5168270812f9f1be7e46a4b435;
- independent Cat original edit ledger court SHA 92afb720dff9ba6a6524901d7363e217ed499f1fdf22bbee0b09c8dabfda0b3e;
- Triceratops original K3 same-Cycles sample physical source court SHA 021786e2bc9bc28d30d717c08eec3d92e64fca1ebc714a451ab85e2988b4882d.

**Independent original-source audit:** [GitHub Actions run 42296438916](https://github.com/merynz/RealSaS-OPT/actions/runs/42296438916) — 18/18 adversarial mathematical/source authority tests and original private Cat + Triceratops source archive join PASS. The exact input bytes were pinned and the report written to private Drive, not public repo.

| Qualified experimental measurement | Result |
|---|---:|
| Cat original local face edges and UV endpoint continuity | 424/424 |
| Cat those local edges with divergent axis mechanics | 59 |
| Cat frozen nonlocal candidates, physically near on fit 0/1 | 497/1,150 |
| Cat those world-near also with original UV endpoint coincidence | 480 |
| Cat original independent UV edit source sample-pixel incidents | 290,979 + 264,554 + 6,048 = **561,581** |
| Cat source edit pair 0/1 projected effect overlap | 2,799 |
| Triceratops original K3 same-sample qualified original source pixels | **1,179,646**, 2 UNKNOWN |
| Triceratops exact physical source palette edit changes | 4,265 / 157,526 / 7,955 |
| Triceratops overlapping multi-source-group original samples | 14,908 |
| Cat direct face-to-physical-edit-pixel relation | **UNKNOWN — not address-mapped** |
| Original authored drawable owner; cross-source same/different labels | **UNKNOWN — not identified** |

The physical results are original-source court facts, **not newly invented samples**. A successful source court does not establish product renderer fidelity, training admission or unseen student generalization.

## IV. What the contract solves, and what remains genuinely open

**Proved and implemented, in declared finite scope:**
- exact source-scope restrictions and refusal to join incompatible observation domains;
- complete uniform Boolean quotient theorem with operation witness certificates;
- partial evidence conflict graph theorem and exact finite partition counterexamples;
- masked-vs-split observational nonidentifiability;
- monotone validity of qualified edit-distinguishing witnesses;
- honest real-source Cat and Triceratops typed evidence reconciliation without owner labels.

**Not proved:** a unique generic graphics partition for any arbitrary character; every source's correct image-space face/UV causal map; alpha/attachment/secondary guarantees; universal 2D graphics runtime equivalence; IRIS unseen learning success. A mathematical theorem about the **correct epistemic contract** is not a theorem that the entire generative inference problem is solved.

**One precise open science gate:** lift original physically changed Cat sample-pixel/ray observations to original **face/corner/UV and operation** addresses using an independently proved sampling and source raster witness. The original PR #351 / #355 physical sample arrays should be reused first. Only if those archived source arrays cannot be mapped to individual source ray-face contributors should we preregister a new source-ray intervention experiment. The next truth label would be *measured operational edit-consequence relation with a validity mask*, never an unqualified source artist drawable group.

**Authority remains:** original owner UNKNOWN; same/different drawable student labels 0; training / materialization / IRIS / TESSA false.

## V. Executable proof assets

- tools/corpus/relations_graphics_proof_contract_v1.py
- tools/corpus/court_relations_graphics_identifiability_real_sources_v1.py
- tests/corpus/test_relations_graphics_proof_contract_v1.py
- tests/corpus/test_relations_graphics_identifiability_real_sources_v1.py
- .github/workflows/relations-graphics-proof-contract-v1.yml

A reviewer can verify the mathematical claims independently by inspecting the tests, exact uniform quotient witnesses, full partition enumeration and the original-source independent court receipts. The proof does not promote any prior SOURCE-ONLY report to student training authority.
