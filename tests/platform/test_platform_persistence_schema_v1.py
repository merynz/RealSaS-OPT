from backend.realsas_platform.persistence.models import Base


def test_required_platform_tables_exist():
    required={"engine_releases","engine_release_stages","subjects","artifact_types","artifacts","artifact_inputs","attempts","attempt_events","attempt_artifacts","executions","execution_artifacts","proofs","qualifications","product_revisions","product_revision_artifacts","promotions","subject_current_revision","render_requests","render_outputs","commands","outbox_events","audit_events"}
    assert required<=set(Base.metadata.tables)

def test_semantic_and_revision_uniqueness_constraints_exist():
    artifacts=Base.metadata.tables["artifacts"]; names={c.name for c in artifacts.constraints if c.name}; assert "uq_artifact_semantic_identity" in names
    revisions=Base.metadata.tables["product_revisions"]; names={c.name for c in revisions.constraints if c.name}; assert {"uq_product_revision_number","uq_product_revision_manifest"}<=names
