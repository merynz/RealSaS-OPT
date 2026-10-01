from __future__ import annotations

from io import BytesIO

from botocore.exceptions import ClientError
import pytest

from backend.realsas_platform.artifacts_s3 import S3ArtifactStoreConfig, S3ContentAddressedStore


class FakeS3:
    def __init__(self):
        self.objects={}

    def head_object(self, *, Bucket, Key):
        try:
            body,meta=self.objects[(Bucket,Key)]
        except KeyError:
            raise ClientError({"Error":{"Code":"NoSuchKey","Message":"missing"},"ResponseMetadata":{"HTTPStatusCode":404}},"HeadObject")
        return {"ContentLength":len(body),"Metadata":dict(meta)}

    def put_object(self, *, Bucket, Key, Body, ContentLength, Metadata):
        assert ContentLength==len(Body)
        self.objects[(Bucket,Key)]=(bytes(Body),dict(Metadata))
        return {}

    def get_object(self, *, Bucket, Key):
        try:
            body,_=self.objects[(Bucket,Key)]
        except KeyError:
            raise ClientError({"Error":{"Code":"NoSuchKey","Message":"missing"},"ResponseMetadata":{"HTTPStatusCode":404}},"GetObject")
        return {"Body":BytesIO(body)}


def test_s3_cas_uses_content_key_and_verifies_bytes():
    fake=FakeS3()
    store=S3ContentAddressedStore(fake,S3ArtifactStoreConfig(bucket="artifacts",key_prefix="realsas"))
    a=store.put_bytes(b"hello")
    b=store.put_bytes(b"hello")
    assert a==b
    assert a.storage_key.endswith(a.content_sha256)
    assert store.get_bytes(a.content_sha256)==b"hello"


def test_s3_cas_detects_existing_corruption():
    fake=FakeS3()
    store=S3ContentAddressedStore(fake,S3ArtifactStoreConfig(bucket="artifacts"))
    obj=store.put_bytes(b"good")
    fake.objects[("artifacts",obj.storage_key)]=(b"bad",{"realsas-sha256":obj.content_sha256})
    with pytest.raises(RuntimeError,match="SIZE_MISMATCH"):
        store.put_bytes(b"good")
