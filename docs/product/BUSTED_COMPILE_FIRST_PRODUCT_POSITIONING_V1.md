# BUSTED — Compile-first product positioning (v1)

## Non-negotiable product distinction

**A playable, controllable puppet is the product.**
An animation is an action that puppet can perform, not the object sold.
A project that merely exports separated 2D layers, a skeleton file, a Spine-compatible rig, or rendered animation clips does **not by itself** meet the BUSTED product contract.

> An animation is something your character does. A puppet is something your character becomes.

**The compiler is responsible for the heavy lifting.** It aims to understand the artwork and assemble a qualified, persistent character:
- geometry and deformable mesh
- mechanical hierarchy, influence, joint relationships and constraints
- skin and appearance fidelity, layers, occlusion, draw order
- animator controls and usable runtime
- verified initial Idle
- editable/inspectable/native artifact with stable identity across sessions

**Studio is optional creative direction**, not the missing manual rigging phase. Studio should enable inspection, fixes, custom posing, target dragging, animation authoring, special clips and export without asking a beginner to assemble the core puppet from scratch.

## Competitor-accurate boundary

God Mode AI's public Spine generator currently describes:
- single-image layer splitting and automatic 2D bone rigging;
- a persistent saved rig and retargeted animations;
- Spine 3.5–4.2 export plus WebM/GIF;
- continued manual editing by importing the exported files into the separate Spine editor.
Their public facts also explicitly state that this Spine rig applies to the **single drawn camera view**, and that new camera angles are not produced by that Spine output. Do not claim that they offer only videos or that their rig is entirely uneditable.

BUSTED does **not** differentiate merely on being editable or using skeletal animation. Its design target is **compiler-closed puppet mechanics** with direct manipulation and 3D-aware consistency beneath a 2D presentation, including multi-view relationships.

Reference pages (public descriptions; competitor behavior beyond documented capabilities not independently verified):
- https://www.godmodeai.co/ai-spine-animation
- https://www.godmodeai.dev/facts

## Pricing and experience

The primary paid result is a **successful compile + verified initial Idle**. Extra animation generation and motion editing should be inexpensive or free within a measured budget. Charging per clip as the central product would hide the puppet value.

User journey: artwork → honest compile telemetry → qualified puppet → real Idle / alive reveal → optional Setup / Motion workspaces → export.

## Required proof before claiming success

No manufactured rig, decorative mannequin or sample animation can substitute for proof. A valid independent/unseen subject must demonstrate source fidelity, persistent structure, direct manipulation and runtime equivalence. Strongest intended demo: user drags the tip of a held sword to a target, wrist/elbow/shoulder/torso/hips/feet respond as permitted, then the character turns while preserving its identity and visible graphics, and an attack can be authored against that target. Two-bone authoring IK is not equivalent to full validated Mechanics execution.

The current product/research pipeline and website must label proposed abilities as **product direction** until compiler authority and unseen tests justify stronger claims. A fit-only reference character is not a generalization result.

## Website copy rules

- Speak about **our models**, **the compiler**, **the puppet**, **Studio**; use **BUSTED** for brand navigation, product identity and essential titles, not every paragraph.
- Lead with **character control / compiler-closed structure**. Editability alone is a supporting property, not the headline.
- Distinguish **Compile (character construction)** from **Studio (optional refinement and motion)**.
- Describe the result as **persistent, directable, living puppet** only as an intended output until qualified, unseen runtime evidence is available.
- Avoid ungrounded comparative attacks; God Mode AI is a legitimate auto-rig + Spine export tool with its own animation workflow.
- After unseen tests pass, place an actual qualified interactive puppet in the public experience, showing mechanics/appearance/controls/Idle as verifiable product behavior.
