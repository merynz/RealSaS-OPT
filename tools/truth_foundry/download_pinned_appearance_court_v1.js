"use strict";

// Called by actions/github-script; no host-installed GitHub CLI is required.
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

module.exports = async function downloadPinnedCourt({ github, context, env = process.env }) {
  const artifactId = Number(env.COURT_ARTIFACT_ID);
  const runId = Number(env.COURT_RUN_ID);
  const expectedDigest = env.COURT_ARTIFACT_DIGEST;
  if (!Number.isSafeInteger(artifactId) || artifactId <= 0 ||
      !Number.isSafeInteger(runId) || runId <= 0 ||
      !/^sha256:[a-f0-9]{64}$/.test(expectedDigest || "")) {
    throw new Error("SOURCE_RAY_COURT_REQUEST_INVALID");
  }
  const { data: metadata } = await github.rest.actions.getArtifact({
    ...context.repo, artifact_id: artifactId,
  });
  if (metadata.id !== artifactId || metadata.name !== env.COURT_ARTIFACT_NAME ||
      metadata.workflow_run?.id !== runId || metadata.expired !== false ||
      metadata.digest !== expectedDigest) {
    throw new Error("SOURCE_RAY_COURT_METADATA_DRIFT");
  }
  const response = await github.rest.actions.downloadArtifact({
    ...context.repo, artifact_id: artifactId, archive_format: "zip",
  });
  const bytes = Buffer.from(response.data);
  const digest = "sha256:" + crypto.createHash("sha256").update(bytes).digest("hex");
  if (digest !== expectedDigest) {
    throw new Error("SOURCE_RAY_COURT_ARCHIVE_BYTES_DRIFT");
  }
  const destination = path.join(env.RUNNER_TEMP, "source-ray-appearance-243.zip");
  fs.writeFileSync(destination, bytes);
  return { path: destination, digest, size_bytes: bytes.length };
};
