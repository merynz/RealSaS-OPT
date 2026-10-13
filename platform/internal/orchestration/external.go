package orchestration

const ExternalStageReadySignalName = "realsas.external-stage-ready.v1"

type ExternalStageReadySignal struct {
    AttemptID      string `json:"attempt_id"`
    StageID        string `json:"stage_id"`
    EvidenceSHA256 string `json:"evidence_sha256"`
}
