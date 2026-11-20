# BUSTED / RSS.vNext — Veri Supervision ve Teacher Teslim Sözleşmesi

**Sürüm:** 2026-11-20 / v1  
**Kapsam:** Veri üretimi, kaynak kanıtı, teacher uygunluğu, eğitim ve ProductProof'a teslim  
**Hedef kitle:** BUSTED veri mühendisliği ve RSS.vNext mimari/Trainer ekibi  
**Durum:** Operasyonel devir dokümanı; **veri seti kabulü, GPU eğitimi, ModelRelease veya ProductPASS beyanı değildir.**

> **Temel talimat:** Her lisanslı karakterden kaynakta gerçekten bulunan geometriyi, görünümü, yüzey ilişkilerini, rig/skin'i, motion ve mekanik davranışı **kayıpsız** çıkarın. Her teacher alanını kaynak kimliğine, tam adresine, geçerlilik maskesine ve bağımsız doğrulama kanıtına bağlayın. Eksik veriyi negatif etiketle doldurmayın. Tek bir nihai **M\*** ve AXIS eklem sırasına sonradan güvenilir biçimde taşınabilen kanıt üretin.

## 0. Mimari bağlam ve sorumluluk sınırı

Ürün: **iki gerçek kaynak FRONT/BACK RGBA** → aynı sahnenin **altı üretilmiş koşullandırma görünümü** → **tek IRIS H(x)** → **TESSA ile tek nihai indexed M\*** → **AXIS rig/controls + MIRA skin** → **Kinematics/Physics sorumluluklarını barındıran Mechanics** → Motion → native **Graphics/2D Presentation** → bağımsız **ProductProof**.

- Gerçek gözlem **yalnız iki kaynak görüntüdür**. Ek altı görünüm *generated conditioning* olarak işaretlenir. Kaynak 3D asset'ten render edilen ek teacher görünümleri, 3D kaynak kanıtı olabilir; **öğrencinin gözlemi** yapılmaz.
- Teacher'ın native mesh'i, kaynak kameraları, rig'i, source face ID'si, skin'i, pose'u ve hak/provenance bilgisi **loss/qualification/evaluation** tarafındadır; H(x) encoder girişine sızdırılmaz.
- **TESSA** bağımsız ikinci bir nihai mesh üretmez: IRIS yüzey tohumunu aynı adreste refine eder. Geometry / rig / skin / UV / Graphics aynı M\* adresine bağlı kalır. Kaynaktaki vertex indeksi, öğrenci M\* vertex indeksi değildir.
- **Graphics** son 2D RGB/alpha/visibility/compositing otoritesidir. SLR/semantik 3D layer ve role kanıtı sunar; otomatik 2D Drawable/cut üretimi mevcut ürün zorunluluğu değildir.
- **Mechanics** içinde Kinematics (FK/IK/LBS/constraint) ve Physics (seçilmiş XPBD secondary cloth/contact) ayrı yürütme sorumluluklarıdır; beşinci neural uzman değildir.
- Compiler/Runtime'ın PASS vermesi, kaynak RGB sadakati, gerçek öğrenme veya ProductRelease PASS anlamına gelmez. ProductProof kapıları birbirini ikame edemez.

## 1. Her asset için ortak L0 kaynak paketi — P0

Her kabul edilen asset için:

| Alan | Üretim ve saklama kuralı |
| --- | --- |
| `canonical_asset_id`, source URI/provider ve türev ailesi | Aynı fiziksel asset'in GLB/FBX/retopo/render varyantları tek soy ilişkisiyle bağlı; split kaçağı yok. |
| Ham kaynak + dış bağımlılıklar | Orijinal `.glb/.gltf/.fbx/.blend` ve texture, material, animation, extra node referansları; immutable byte SHA-256. |
| Lisans ve kullanım kanıtı | Kaynak/asset seviyesinde lisans, yazar/attribution, versiyon ve AI-train yasağı kontrolü. Açık izin yoksa `QUARANTINE/REJECT`; `NoAI`, NC, ND, research-only gibi engeller görmezden gelinmez. |
| Source scene | Orijinal mesh/nodes, scene graph, object/world/local transforms, birim, handedness, ground/root, rest/bind/frame0 ayrımı. |
| Normalizasyon | Source → canonical matris ve tersi; geometri, rig, skin, görüntü ve animasyonda aynı transform. |
| Kamera ve iki gerçek gözlem | FRONT/BACK RGBA, image-space conventions, exact projection/intrinsics/extrinsics, background/alpha/color-space policy, camera SHA. Kaynak kamera doğrulanamıyorsa `CAMERA_UNKNOWN` ve ilgili ray maskeleri kapalı. |
| Hakem renderları | Kaynaktan aynı pose/kamera ile bağımsız yeniden render, source/import parity ve hash; yalnız boş olmayan PNG üretmek kabul değildir. |
| Metaveri | Source importer, Blender/render motoru sürümü, producer commit/SHA, seed, timestamp, tensor dtype/shape/hash, status/mask, ret/skip reason. |

Öncelik: **RAW source asla overwrite edilmez**; normalized/teacher ve renderlar türevdir. Eksik rig/skin/animasyon nedeniyle bir geometri asset'i çöpe gitmez; yalnız ilgili teacher ailesi devre dışı kalır.

## 2. Geometry / IRIS / TESSA — P0/P1

### 2.1 Native yüzey ve kaynak adresleri

İlk çıkartılacak veri: orijinal indexed vertices/faces, normal/tangent, UV/material, connected-component, geometrik topoloji (watertight / OPEN / non-manifold / unknown), gerçek açık hem/edge ve yüzey örnekleri. **Face ID + barycentric** orijinal kaynak topolojisine referans verir; XYZ-nearest tek başına gerçek eşleşme oluşturmaz. Aynı XYZ'de duran gövde ve pelerin **farklı layer/face** olarak kalır.

`SeededTESSASourceTeacherV1` alanları (mevcut trainer kodu):

| Kaynak teacher alanı | Şekil ve anlam | Kural |
| --- | --- | --- |
| `source_xyz` | `[M,3]` source-native yüzey XYZ | Kaynak 3D'den ölçülmüş. |
| `source_normals` | `[M,3]` yüzey normalleri | Orientasyon/doğruluk maskeli. |
| `source_point_mask` | `[M]` bool | Gerçek geçerli nokta/normal. |
| `source_importance` | `[M]`, `[0,1]` | Native curvature, multi-pose strain veya kaynak odaklı yapısal önem. |
| `source_importance_mask` | `[M]` bool | Öğretmen ölçümü yoksa masked. |
| `source_component_ids` | `[M]` | Native bağlantılı bileşenler; öğrenci slot ID'si değildir. |
| `source_layer_ids` | `[M]` | Native materyal/part/layer kaynaklı doğrulanmış katman. |
| `native_face_layer_samples` | Source vertices/faces, `face_layer_ids`, `face_component_ids`, `sample_face_ids`, `sample_barycentric` | Native face + barycentric tekrar oynatıldığında aynı XYZ/layer/component çıkmalı. |
| `native_open_boundary` | Açık edge/hem + source layer/face ilişkisi | Gerçek OPEN component; kapalı yüzeye sahte hem çizilmez. |
| `source_boundary_xyz`, `source_boundary_mask` | `[K,3]`, `[K]` | Gerçek açık kenar truth'u. |
| `native_dynamic_importance` | Native pose/strain kaynaklı source face önem/risk | Kaynak motion yoksa bu kanaldan sahte risk üretilmez. |
| `seeded_tessa_source_face_budget` | Kaynak ölçekte gerçekçi yüz bütçesi | Kaynak topoloji birebir ground-truth token aksiyomu değildir. |

**SDF politikası:** `source_udf_target [B,Q]` + `source_udf_mask` açık/kapalı yüzeylerde kullanılabilir. `source_signed_sdf_target`, `source_inside_sign_target` ve maskeleri **yalnızca independently qualified closed solid / tutarlı boolean assembly** için pozitif olabilir. Gerçek açık kumaş/kanat/pelerin üzerinde inside/outside uydurulmaz. Component başına alan için ayrıca `SourceNativeComponentUDFFitV1`: `original_query_xyz [Q,3]`, `source_component_udf [C,Q]`, `source_component_mask [C,Q]`, `source_component_open_kind [C]`, `native_component_inventory_sha256`, `independently_complete_inventory`.

**TESSA üretim hedefi:** sürekli rest surface, normals, OPEN boundary, mesh quality, source curvature/deformation importance, budget. Kaynaktaki triangulation ve retopo edit sırası ayrı token CE mecburiyeti değildir. Seeded production profilinde eski bağımsız topology-token/motion-preference loss'larını **zorunlu teacher** saymayın. Deformation/pose-strain ve layer-penetration ancak ek kaynak/transport kanıtı olduğunda açılır.

## 3. IRIS SLR / Semantic — P1

Üretilmesi istenenler:

| Alan | Şekil / kanıt |
| --- | --- |
| `semantic_patch_target`, `semantic_patch_mask` | `[B,2,T,K]`: iki orijinal görüntüde bilinen patch-ekseni; her eksen ayrı bool maskeli. |
| `source_surface_role_indices`, `source_surface_role_mask` | `[B,N]`: **aynı H(x) XYZ query** üzerinde native/renderer kaynaklı semantik surface role. |
| `source_region_pair_indices` | `[B,P,2]` source-bound 3D query çiftleri. |
| `source_region_pair_is_same`, `source_region_pair_mask` | `[B,P]` aynı native region mı ve mask; farklı joint/constraint demek değildir. |
| Face-level native evidence | Material/part/layer/component, open-vs-closed, gerçek authored metadata; kaynağı yoksa UNKNOWN. |

SAM/manual segmentation destekleyici olabilir; **2D maskeden otomatik 3D Drawable, attachment, kemik, cloth material veya bütün gizli bölge gerçeği üretilmez.** `SAME_DRAWABLE / CUT_BOUNDARY / DIRECTED_ATTACHMENT` üçlü affinity eski araştırma alanıdır. **2026-11-19 ürün kararıyla default product typed-relations emisyonu devre dışıdır** (`PRODUCT_NO_TYPED_RELATIONS`). Veri ekibinin P0/P1 mandatory teslimi değildir. Kaynakta gerçek constraint/anchor varsa bunlar AXIS/Mechanics öğretmeni olarak ayrı alınır.

## 4. Graphics / 2D Presentation — P1 ve P2

Graphics için kaynak RGB tek başına yetmez. **Görünürlüğün kim tarafından üretildiği**, alfa ve ray stacking ölçülmeli.

| Kanıt | İstenen ölçümler |
| --- | --- |
| Orijinal 2-view image | Kaynak RGBA, matting/alpha, renk uzayı, premultiplication ve camera. |
| Source camera/raster | Pixel index → source first-hit face ID + barycentric, XYZ, normals, depth, coverage ve validity. |
| `source_graphics_rgb_target` | `[B,Q,3]`: yalnız source-qualified **opaque first-hit** RGB. |
| `source_graphics_view_directions` | `[B,Q,3]`: aynı XYZ ve tam kaynak kamera ray yönleri. |
| `source_graphics_rgb_mask` | `[B,Q]`: doğru/visible/opaque/first-hit ölçümleri; hidden/translucent/ambiguous masked. |
| Stage1/Stage2 source ray | `RealSaSJointTargetV1`: RGB white, alpha, role indices (`-1` UNKNOWN), `qualified_rgb_mask`, `qualified_mask_mask`, `qualified_role_mask`, source 3D truth binding, layer/stage. |
| Full-hit raster/ray | First/next-hit face IDs, depth/ties, material alpha, coverage alpha, transmittance T, blend law, ordered contributions, `UNKNOWN` mask, renderer/camera hash. |
| UV/material/texture | Orijinal texture bytes, texel/face mapping, color and alpha source provenance, UV seams ve authored material conventions. |
| Motion-linked source Graphics | Gerçek kaynak clip/pose ile aynı sealed kamera, zaman, front owner, RGBA/alpha, yeniden görünür olan bölgeler ve source-authoritative vs inferred ayrımı. |

**Stage 3 açıklığı:** Trainer'daki Stage1/2 joint ray loss'ları var; gerçek **nihai M\*** differentiable triangle rasterizer üzerinden Stage3 appearance/depth/normal training kapısı, kaynakta eksiksiz doğrulanmış sayılmamalı. Veri ekibi Stage3'e yönelik native source RGB/alpha/depth/normal/UV/ray truth'u saklamalı; bir volume NeRF ray çıktısını gerçek mesh pixel truth yerine koymamalı.

Kaynakta görünmeyen bir yüzeyin GT rengini yalnız source-native 3D material/texture veya doğrulanabilir source render kanıtı sağlıyorsa etiketleyin. Diffusion'dan gelen tahmin **pseudo completion** olarak ayrı saklanır; kaynak truth değildir. Yarı saydam ön yüzeyin bilinen rengi, arkasındaki `UNKNOWN` varlığını silmez.

## 5. AXIS causal rig / MIRA skin — P1

**Kritik: gerçek native rig ve skin, kaynak animasyon içermese bile teacher olabilir.**

### 5.1 Native rig ve AXIS programı

Kaynak armature/joint graph, deform vs control bones, local/world/rest/bind/inverse-bind, roots/parents, gerçek aux chains, source joint-order, authored FK/IK/constraint/socket inventory, vertex groups, mesh/skin binding ve validite kaydı oluşturun. Öğretmen AXIS V6 causal grammar/tokens, yalnız kaynak doğruladığı joint/parent/structural chain bilgisini zorunlu üretir; `CONTROL/CONSTRAINT/ANCHOR` etkin öğretimi **ayrı authored law/intervention** gerektirir. Anatomik bone adları veya part segmentation tek başına operator truth değildir.

### 5.2 Native skin ve M\* transport

| Alan | Şekil / sözleşme |
| --- | --- |
| `axis_teacher_sequence` + admission key | `AXISSequenceV6`, source-native joint/parent/eligible chain, pinned frozen FSQ codec. |
| `native_skin` | `[N,J]`: source vertex ve native joint sırasındaki gerçek weights. |
| `native_skin_joint_order_sha256` | AXIS teacher sırasıyla **bağımsız** kaynak eklem eşleşmesi. |
| `candidate_native_face_ids` | `[V]`: final candidate vertex için **independently admitted original source face**. |
| `candidate_barycentric` | `[V,3]`: aynı source face üzerine source-supported barycentric. |
| `candidate_mstar_topology_address_sha256` | Aynı final M\* ve M0 parent/topology kimliği; teacher kendi hayali M\* vertex ID'si yaratmaz. |
| `axis_chain_joint_ids` | `[J]` source-native chain partition, `-1` UNKNOWN. |
| `axis_chain_vertex_mask` | `[1,V]` chain/skin supervision geçerliliği. |

Geometric nearest-neighbor ile student mesh'ten eşleşme icat edilmez; true native face support + barycentric, doğru joint order ve source rest reprojection tolerance ayrı ayrı doğrulanır. MIRA skin, final M\* üzerinde `[V,J]` olarak üretilir. Chain loss, aynı skin üzerinde `P(chain|v) × P(joint|chain,v)` ayrışımıdır; yeni skin modeli değildir. Codec önce source-native data ile eğitilir ve dondurulur; rastgele FSQ kodları gerçek deformasyon anlamına gelmez.

## 6. Motion, posed surfaces, Kinematics ve Physics — P2

Kaynakta gerçek animation varsa native clip manifesti; clip/track, exact sample time, joint local/global matrices, morph targets, root/object motion, rest ile frame0 farkı ve **gözlenmiş posed vertex** ölçümlerini üretin.

| Teacher alanı | Şekil / şart |
| --- | --- |
| `axis_pose_joint_skin_matrices` | `[1,P,J,4,4]` native same-AXIS-order skin matrices. |
| `axis_pose_rest_surface_samples` | `[S,3]` source rest. |
| `axis_pose_surface_samples` | `[P,S,3]` source LBS-qualified pose. |
| `axis_pose_valid_mask` | `[P]`: poz örneklerinin bağımsız geçerliliği. |
| `source_candidate_posed` / mask | `[1,P,V,3]` + `[1,P,V]`, source→M\* exact reindex doğrulanmışsa. |
| `native_pose_transport` | Native skin + source face/barycentric + pose matrices + per-pose correspondence + receipt. |

Animasyon declaration veya clip adı, LBS ile açıklanabilir deformasyon teacher'ı olduğunun kanıtı değildir. Source rest→posed LBS independently replay edilebilmeli; source authored skinned/morph/simulation farkları ayrı etiketlenmeli. `source_skin` ile `posed_surface` **farklı yetki seviyeleridir**.

### Kinematics / attachments

Authored IK constraints, hinge/limits, sockets, held/holder faces ve local grip transforms; kontroller üzerinde pozitif/negatif ve sham perturbation, root-only kontrol, relative transform, anchor residual ve hareket serbestliği ölçülebilmeli. Yakınlık, aynı parent ile co-motion veya aynı body-part etiketi tutuşma/kısıt truth'u değildir. Native face/anchor adresi ve valid mask zorunlu.

### Physics / XPBD

Gerçek OPEN garment/wing membrane, source hem, pin vertices ve host joint chain, collision/contact data; fizik native varsa timestep, init/reset state, stiffness/compliance, damping/mass provenance, external force, secondary displacement/decay ve tekrar oynatma kanıtı saklanır. Bunlar kaynaktan gelmiyorsa **Physics policy candidate** olabilir, *observed physical truth* olamaz. Source-open pin kabulü için bağımsız host-only intervention ve Compiler'ın final M\* transport receipt'i gereklidir. Multi-host cloth, contact self-collision ve gerçek materyal kestirimi ayrıca doğrulanır.

### Müdahale/response paketi (kaynakça uygun olduğunda)

İleride Mechanics, Graphics ve ProductProof için tam kaynak locus'u `(source_hash, topology_hash, face, barycentric)`, action law/amplitudes, timestamps, source trajectory/velocity, local strain/area/bending, relative affine transport, oriented gap/slip, rank/nullspace, contact impulse (yalnız gerçek physics source), full-ray compositing/occlusion ve response validity/observability saklayın. Bir clip fiziksel yük/kuvveti kendiliğinden açıklamaz.

## 7. Her asset için kabul seviyesi ve eksik alan politikası

| Seviye | Mevcut gerçek kaynak | İzin verilen teacher ailesi | İzin verilmeyen iddia |
| --- | --- | --- | --- |
| `GEO` | Hakları/açıklığı doğrulanmış native 3D mesh + 2 source RGBA | IRIS UDF, qualified SDF, TESSA source geometry/boundary | Rig/skin/physics uydurmak |
| `GEO_GRAPHICS_SEM` | Yukarıdakiler + verified material/UV/semantic/camera/ray | SLR patch/roles, Graphics visible/RGB/alpha/rays | 2D segmentation→attachment, generated view→observed truth |
| `RIG_SKIN` | Yukarıdakiler + native joint/parents/bind/weights, animasyon şart değil | AXIS structural sequence, MIRA skin, qualified chain/FSQ | Posed surface/contact/physical laws |
| `POSE` | `RIG_SKIN` + replayed native LBS motion | Source posed-surface/strain, clip evaluation, moving Graphics | Physics/contact her koşulda gerçek sanmak |
| `INTERVENTION_PHYSICS` | Source authored anchors/constraints/dynamics + measured interventions | Kinematics/attachment/Physics eligibility, independent behavior Proof | Kaynakta olmayan force/mass/pin'i GT yapmak |

**Her supervision ailesi için 3 yönlü etiket:** `QUALIFIED`, `UNKNOWN/UNAVAILABLE`, `UNSUPPORTED` ve sebep; ayrıca `MASKED` örnek/pixel/face listeleri. `UNKNOWN` hiçbir zaman true negative veya all-zero teacher değildir. Kaynak hak veya source fidelity kaybı varsa uygun aile `FAIL/QUARANTINE`.

## 8. Önerilen asset teslim dizini

Bu dizin **çalışma klasörü önerisidir**; mevcut Python dataclass'larının bütününün otomatik olarak tanınan tek bir wire formatı olduğu iddia edilmez. Resmi schema/admission serileştirmesi işin consumer sınırında ayrıca yapılır.

```text
asset_id/
├── source/
│   ├── original.glb | original.fbx | original.blend
│   ├── dependencies/
│   ├── license_provenance.json
│   └── source_identity.json
├── observations/
│   ├── front.rgba.png
│   ├── back.rgba.png
│   ├── source_cameras.json
│   ├── source_views.npz
│   └── source_ray_evidence.npz
├── geometry/
│   ├── native_mesh.npz
│   ├── surface_samples.npz
│   ├── sdf_udf_queries.npz
│   ├── component_inventory.json
│   ├── native_face_layers.npz
│   └── open_boundaries.npz
├── semantics/
│   ├── patch_targets.npz
│   ├── surface_roles.npz
│   └── region_pairs.npz
├── graphics/
│   ├── source_rgb_alpha_truth.npz
│   ├── uv_material_truth.json
│   └── ray_stacks.npz
├── rig_skin/
│   ├── native_rig.json
│   ├── native_skin.npz
│   ├── source_chain_labels.npz
│   ├── axis_teacher.npz
│   └── source_face_transport.npz
├── motion/
│   ├── native_clips.json
│   ├── joint_matrices.npz
│   ├── posed_surfaces.npz
│   └── pose_validity.npz
├── mechanics/
│   ├── source_constraints.json
│   ├── attachment_witnesses.json
│   └── physical_response.npz
└── admission/
    ├── qualification_masks.npz
    ├── source_replay_report.json
    ├── teacher_eligibility.json
    └── manifest.json
```

`manifest.json` en az: `canonical_asset_id`, `source_asset_sha256`, `dataset_manifest_sha256` (dataset sabitlendiğinde), license/source provenance, source importer/renderer version ve SHA, geometry source/topology hash, normalization/camera hash, layer namespace hash, AXIS joint order hash, M\* transport address hash (sonradan), per-tensor dtype/shape/hash, valid mask/status + reason, generated-view conditioning checkpoint provenance, family-wise `QUALIFIED/UNAVAILABLE/UNSUPPORTED` ve producer git commit taşır.

**Kural:** Kaynak yüz indeksleri native source namespace'te yaşar. M\* indeksleri **model derledikten sonra** Compiler/teacher transport namespace'te üretilir. Veri ekibi öğrenci M\*'nin tahmin edeceği vertex sayısını/face order'ını peşinen uyduramaz.

## 9. Üretim çalışma sırası / kabul kontrolleri

- [ ] **P0 — Kaynak kabulü:** asset-hash, objektif lisans/NoAI kontrolü, dependency closure, source scene, bind/rest/frame0, ortak normalization ve FRONT/BACK kamera/observation doğrula.
- [ ] **P0 — Native geometri:** indexed mesh, component/layer, gerçek OPEN hem, source face+barycentric, mesh normals; SDF için watertight signed-proof, OPEN için component UDF.
- [ ] **P1 — Source-visible Graphics:** RGBA/alpha, exact source camera, opaque first-hit RGB, face-owner/depth, UV/material, full ray ve `UNKNOWN` masks. Source render/import parity.
- [ ] **P1 — Semantic:** source patch axes, 3D layer/role, same-region pairs; maskeleri ve kaynak yüz adreslerini yeniden oynat.
- [ ] **P1 — AXIS/MIRA:** native hierarchy, rest/bind/inverse binds, skin `[N,J]`, chain evidence, AXIS teacher sequence, frozen MIRA codec; animasyonsuz skin'i kabul et.
- [ ] **P1 — Final M\* transport:** original source face+barycentric ile tahmin edilen M\* ve AXIS order'a bağımsız adres doğrulaması; student-nearest shortcut yok.
- [ ] **P2 — Motion:** native clips + exact timestamps/pose matrices + actual posed source mesh + LBS replay ve moving-source Graphics.
- [ ] **P2 — Mechanics/XPBD:** authored constraints, anchors, pins, intervention, source physical response ve bağımsız runtime residuals; kanıt yoksa kapalı.
- [ ] **Bütün seviyeler:** Her eligible loss için tensor SHA/mask/source parent/registry admission; teacher-to-model forward leakage denetimi.
- [ ] **Train/Val/LOFO:** aynı kaynak ve türev ailesi split içinde tutulur, creator/rig/topology/material lineage kaçağı raporlanır; unseen kimlikler eğitimden önce sabitlenir.
- [ ] **Bilimsel kabul:** aynı source-bound batch'te loss gerçekten tüketiliyor mu ve gradient ilgili head'e gidiyor mu; CPU synthetic correctness ≠ real-corpus/GPU/held-out quality ≠ ProductRevision.

**Öncelik tavsiyesi:** İlk önce P0 ve kaynak–face–barycentric kimliğini kapat; hemen ardından **Graphics RGB/alpha/visibility** ve **AXIS/MIRA native rig/skin**. Kaynak animasyonu/physics olmayan asset'leri atmayın. P2 için özellikle wing, cloth, cape, tail, armor, held equipment ve crossing-limb örnekleri coverage stratumu oluşturun.

## 10. Bugünkü Trainer/Compiler bağlantı durumu — kesinlik uyarısı

- `training_profile=PRODUCT_SEEDED_IRIS_TESSA_V1`: seed TESSA source-native rest/normal/structural loss, aynı IRIS H(x); optional Semantic, Graphics RGB/joint ray, source UDF, AXIS/MIRA, source skin/chain/pose/attachment bağımsız ağırlık + trusted receipt registry ile açılıyor.
- `PRODUCT_COMPONENT_UDF_V1`: open/multiple component UDF source-only ayrı eğitim profili, seeded profile içinde qualified opt-in de mümkün.
- `product_semantic_weights.typed_relations` mevcut araştırma alanı olarak kodda bulunsa bile **product default retired**; zorunlu veri üretme hakkı vermez.
- Tüm opsiyonel loss ağırlıkları `0` olabilir: kodda tanımlı olmak **o kaynağın hâlihazırda corpus'ta mevcut, üretilmiş veya eğitimde aktif** olduğunu göstermez.
- `NativeCandidateSkinTransport` yalnız native skin ve face support kanıtını taşır; `NativeCandidatePoseTransport` ayrıca gerçek LBS pose ister. Birinin receipt'i diğerini karşılamaz.
- Graphics Stage1/2 ray loss vardır; mesh Stage3, native post-Physics render parity ve ProductProof bütün hareketli karakterlerde kaynak sadakati olarak **henüz kabul edilmiş değildir**.
- `product.proof` tek ProductPASS otoritesidir. Native edit raster, temporal inventory ve host-replayed sample kısmi execution kanıtlarıdır; source artwork fidelity, physical contact ve release yetkisini tek başına kapatmaz.

## 11. Referanslar (aktif kod ve mimari kararları)

1. [Trainer batch / profile / trusted registry](https://github.com/merynz/RealSaS-OPT/blob/main/models/rss_vnext/training_engine.py)
2. [Seeded TESSA source teacher](https://github.com/merynz/RealSaS-OPT/blob/main/models/rss_vnext/training_seeded_tessa_v1.py)
3. [Source native face+barycentric layer witness](https://github.com/merynz/RealSaS-OPT/blob/main/models/rss_vnext/native_face_layer_truth_v1.py)
4. [Source component UDF](https://github.com/merynz/RealSaS-OPT/blob/main/models/rss_vnext/component_conditioned_udf_v1.py)
5. [Semantic source loss](https://github.com/merynz/RealSaS-OPT/blob/main/models/rss_vnext/product_semantic_supervision_v1.py)
6. [Graphics source RGB and joint ray loss](https://github.com/merynz/RealSaS-OPT/blob/main/models/rss_vnext/product_graphics_supervision_v1.py)
7. [AXIS/MIRA seeded source training](https://github.com/merynz/RealSaS-OPT/blob/main/models/rss_vnext/product_axis_mira_seeded_training_v1.py)
8. [Source-native skin transport without animation](https://github.com/merynz/RealSaS-OPT/blob/main/models/rss_vnext/native_skin_transport_v1.py)
9. [Source-native pose transport](https://github.com/merynz/RealSaS-OPT/blob/main/models/rss_vnext/native_pose_transport_v1.py)
10. [Mechanical semantics and Product typed-relations retirement (2026-11-19)](https://github.com/merynz/RealSaS-OPT/blob/main/docs/research/RSS_VNEXT_MECHANICAL_SEMANTICS_CLOSURE_20261119.md)
11. [Source qualified OPEN pin / XPBD](https://github.com/merynz/RealSaS-OPT/blob/main/docs/rss_vnext/source_qualified_open_surface_pin_xpbd_contract.md)
12. [Source-first truth foundry primitives](https://github.com/merynz/RealSaS-OPT/blob/main/docs/research/truth_foundry/TRUTH_FOUNDRY_V1.md)
13. [Loss inventory (2026-11-17; dated snapshot, later decisions may supersede)](https://github.com/merynz/RealSaS-OPT/blob/main/models/rss_vnext/training_supervision_inventory_v1.json)

---

**Devir cümlesi:** *Öğretmen paketlerimiz, modeli doğru cevaba zorla götüren gizli input değildir; kaynakta gerçekten ölçülmüş cevapları, aynı tek puppet'ın geometri–rig–skin–Graphics ilişkilerini öğrenebilmesi ve bağımsız doğrulayabilmesi için kayıpsız, maskeli ve adresli sunar.*