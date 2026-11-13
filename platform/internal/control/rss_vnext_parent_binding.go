package control

import (
	"errors"
	"fmt"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/modelselection"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
)

// rssVNextMeshParentBinding binds the source and learned checkpoint to the
// immutable Attempt model selection. Mesh never loads or selects a second
// ModelRelease. This is the missing Go->Python Shape v3/Mesh v2 handshake.
//
// The Shape fanout and inference receipt are still verified independently by
// the Python Mesh court. This binding is NOT a grant of source 3D truth,
// CanonicalMesh, Rig, Skin, or ProductRevision authority.
func rssVNextMeshParentBinding(
	refs []orchestration.ArtifactRef,
	selected []modelselection.Selection,
) (map[string]any, error) {
	var source *orchestration.ArtifactRef
	for i := range refs {
		row := &refs[i]
		if row.ArtifactType != "RealSaS.RSSvNextTwoViewSource" {
			continue
		}
		if row.SchemaVersion != "v1" || source != nil {
			return nil, errors.New("RSS_VNEXT_MESH_SOURCE_ARTIFACT_CARDINALITY_OR_ABI")
		}
		if err := semantic.ValidateSHA256(row.ContentSHA256); err != nil {
			return nil, fmt.Errorf("RSS_VNEXT_MESH_SOURCE_SHA_INVALID: %w", err)
		}
		source = row
	}
	if source == nil {
		return nil, errors.New("RSS_VNEXT_MESH_SOURCE_ARTIFACT_REQUIRED")
	}
	// No historical IRIS/TESSA/AXIS/MIRA family may be silently selected.
	if len(selected) != 1 || selected[0].Family != capability.RSSvNextModelFamily {
		return nil, errors.New("RSS_VNEXT_MESH_SINGLE_ATTEMPT_MODEL_REQUIRED")
	}
	model := selected[0]
	if model.ArchitectureID != capability.RSSvNextArchitectureID {
		return nil, errors.New("RSS_VNEXT_MESH_UNIFIED_ARCHITECTURE_DRIFT")
	}
	if model.ArtifactType != "RealSaS.RSSvNextUnifiedInferenceCheckpoint" ||
		(model.ArtifactSchemaVersion != "v1" && model.ArtifactSchemaVersion != "v2") {
		return nil, errors.New("RSS_VNEXT_MESH_CHECKPOINT_ABI_DRIFT")
	}
	if err := semantic.ValidateSHA256(model.CheckpointContentSHA256); err != nil {
		return nil, fmt.Errorf("RSS_VNEXT_MESH_CHECKPOINT_SHA_INVALID: %w", err)
	}
	if len(model.SourceCodeSHA) != 40 {
		return nil, errors.New("RSS_VNEXT_MESH_SOURCE_CODE_SHA_INVALID")
	}
	// Exact cross-language JSON contract in platform/internal/semantic/hash.go
	// and compiler/.../platform_worker/model_inputs.py. Do not approximate
	// release behavior as checkpoint hash alone.
	identity := map[string]any{
		"schema":                    "RealSaS.ModelReleaseBehaviorIdentity.v1",
		"family":                    model.Family,
		"model_release_id":          model.ModelReleaseID.String(),
		"architecture_id":           model.ArchitectureID,
		"model_engine_release_id":   model.ModelEngineReleaseID.String(),
		"source_code_sha":           model.SourceCodeSHA,
		"checkpoint_artifact_id":    model.CheckpointArtifactID.String(),
		"checkpoint_content_sha256": model.CheckpointContentSHA256,
	}
	behaviorSHA, err := semantic.JSONSHA256(identity)
	if err != nil {
		return nil, fmt.Errorf("RSS_VNEXT_MESH_MODEL_IDENTITY_HASH_FAILED: %w", err)
	}
	return map[string]any{
		"schema":                        "RealSaS.RSSvNext.ShapeParentBinding.v1",
		"source_content_sha256":         source.ContentSHA256,
		"checkpoint_sha256":             model.CheckpointContentSHA256,
		"model_release_behavior_sha256": behaviorSHA,
	}, nil
}
