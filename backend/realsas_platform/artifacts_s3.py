from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from typing import Any

import boto3
from botocore.exceptions import ClientError

from .artifacts import StoredObject


@dataclass(frozen=True)
class S3ArtifactStoreConfig:
    bucket: str
    endpoint_url: str | None = None
    region_name: str | None = None
    key_prefix: str = ""

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError("S3 artifact bucket is required")
        if self.key_prefix.startswith("/"):
            raise ValueError("S3 key_prefix must be relative")


class S3ContentAddressedStore:
    """Provider-neutral S3 API content-addressed artifact store.

    Object names are immutable content identities. Registration code still owns
    the DB transaction that makes an uploaded object visible to the platform.
    """

    def __init__(self, client: Any, config: S3ArtifactStoreConfig) -> None:
        self.client = client
        self.config = config

    @classmethod
    def from_config(cls, config: S3ArtifactStoreConfig) -> "S3ContentAddressedStore":
        client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region_name,
        )
        return cls(client, config)

    def key_for(self, content_sha256: str) -> str:
        if len(content_sha256) != 64 or any(c not in "0123456789abcdef" for c in content_sha256):
            raise ValueError("invalid sha256")
        base = f"cas/sha256/{content_sha256[:2]}/{content_sha256[2:4]}/{content_sha256}"
        prefix = self.config.key_prefix.strip("/")
        return f"{prefix}/{base}" if prefix else base

    def put_bytes(self, data: bytes) -> StoredObject:
        digest = sha256(data).hexdigest()
        key = self.key_for(digest)
        try:
            existing = self.client.head_object(Bucket=self.config.bucket, Key=key)
        except ClientError as exc:
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if status != 404 and code not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            if int(existing.get("ContentLength", -1)) != len(data):
                raise RuntimeError("S3_CAS_EXISTING_SIZE_MISMATCH")
            metadata = {str(k).lower(): str(v) for k, v in dict(existing.get("Metadata") or {}).items()}
            if metadata.get("realsas-sha256") not in {None, digest}:
                raise RuntimeError("S3_CAS_EXISTING_METADATA_HASH_MISMATCH")
            if not self.verify(digest):
                raise RuntimeError("S3_CAS_EXISTING_CONTENT_HASH_MISMATCH")
            return StoredObject(digest, key, len(data))

        self.client.put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=data,
            ContentLength=len(data),
            Metadata={"realsas-sha256": digest},
        )
        if not self.verify(digest):
            raise RuntimeError("S3_CAS_POST_WRITE_VERIFICATION_FAILED")
        return StoredObject(digest, key, len(data))

    def get_bytes(self, content_sha256: str) -> bytes:
        key = self.key_for(content_sha256)
        response = self.client.get_object(Bucket=self.config.bucket, Key=key)
        body = response["Body"].read()
        if sha256(body).hexdigest() != content_sha256:
            raise RuntimeError("S3_CAS_CONTENT_HASH_MISMATCH")
        return body

    def verify(self, content_sha256: str) -> bool:
        try:
            self.get_bytes(content_sha256)
        except (ClientError, RuntimeError):
            return False
        return True
