package resolver

import (
    "context"
    "testing"

    "github.com/merynz/RealSaS-OPT/platform/internal/domain"
)

func stageSemantic(plan Plan, stageID string) string {
    for _, row := range plan.Stages {
        if row.StageID == stageID { return row.ExpectedSemanticSHA256 }
    }
    return ""
}

func containsStage(values []string, wanted string) bool {
    for _, value := range values { if value == wanted { return true } }
    return false
}

func TestExternalEvidenceChangesExactConsumerAndDescendantsButNotAncestors(t *testing.T) {
    g := resolverGraph(t)
    catalog := newMemoryCatalog()
    root := []domain.ArtifactInputIdentity{{
        Role: "subject:source", ArtifactType: "RealSaS.SourceImage", SemanticSHA256: hex64("a"),
    }}
    versions := resolverVersions(g, hex64("1"))
    baseline, err := ResolveWithExternal(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, versions, nil)
    if err != nil { t.Fatal(err) }
    catalog.admit(baseline)

    external := map[string][]domain.ArtifactInputIdentity{
        "10_IRIS_FIT": {{
            Role: "checkpoint", ArtifactType: "RealSaS.ExternalModelCheckpoint", SemanticSHA256: hex64("e"),
        }},
    }
    changed, err := ResolveWithExternal(context.Background(), g, catalog, "46_PRODUCT_CLOSURE_SEAL", root, versions, external)
    if err != nil { t.Fatal(err) }

    for _, ancestor := range []string{"01_SOURCE_BYTES_SEALED", "02_SOURCE_LICENSE_PROVENANCE", "09_IRIS_FIT_PREREGISTERED"} {
        if stageSemantic(baseline, ancestor) != stageSemantic(changed, ancestor) {
            t.Fatalf("external Stage10 evidence invalidated ancestor %s", ancestor)
        }
        if containsStage(changed.ExecuteStageIDs(), ancestor) {
            t.Fatalf("qualified ancestor re-executed: %s", ancestor)
        }
    }
    if stageSemantic(baseline, "10_IRIS_FIT") == stageSemantic(changed, "10_IRIS_FIT") {
        t.Fatal("Stage10 semantic identity did not include external evidence")
    }
    if stageSemantic(baseline, "11_IRIS_CHECKPOINT_SEALED") == stageSemantic(changed, "11_IRIS_CHECKPOINT_SEALED") {
        t.Fatal("Stage10 external evidence did not invalidate descendant Stage11")
    }
    if !containsStage(changed.ExecuteStageIDs(), "10_IRIS_FIT") || !containsStage(changed.ExecuteStageIDs(), "11_IRIS_CHECKPOINT_SEALED") {
        t.Fatalf("changed execution scope missing Stage10/11: %v", changed.ExecuteStageIDs())
    }
}

func TestChangingExternalCheckpointChangesStage10SemanticIdentity(t *testing.T) {
    g := resolverGraph(t)
    root := []domain.ArtifactInputIdentity{{Role:"subject:source", ArtifactType:"RealSaS.SourceImage", SemanticSHA256:hex64("a")}}
    versions := resolverVersions(g, hex64("1"))
    one := map[string][]domain.ArtifactInputIdentity{"10_IRIS_FIT": {{Role:"checkpoint",ArtifactType:"RealSaS.ExternalModelCheckpoint",SemanticSHA256:hex64("c")}}}
    two := map[string][]domain.ArtifactInputIdentity{"10_IRIS_FIT": {{Role:"checkpoint",ArtifactType:"RealSaS.ExternalModelCheckpoint",SemanticSHA256:hex64("d")}}}
    p1,err:=ResolveWithExternal(context.Background(),g,newMemoryCatalog(),"11_IRIS_CHECKPOINT_SEALED",root,versions,one);if err!=nil{t.Fatal(err)}
    p2,err:=ResolveWithExternal(context.Background(),g,newMemoryCatalog(),"11_IRIS_CHECKPOINT_SEALED",root,versions,two);if err!=nil{t.Fatal(err)}
    if stageSemantic(p1,"10_IRIS_FIT")==stageSemantic(p2,"10_IRIS_FIT") { t.Fatal("checkpoint SHA drift was invisible") }
    if stageSemantic(p1,"11_IRIS_CHECKPOINT_SEALED")==stageSemantic(p2,"11_IRIS_CHECKPOINT_SEALED") { t.Fatal("descendant SHA drift was invisible") }
}
