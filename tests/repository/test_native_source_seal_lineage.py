from copy import deepcopy
import pytest

from tools import verify_native_runtime_source_seal_v2 as seal


def chain():
    return seal._load_json(seal.BASE), [seal._load_json(path) for path in seal.EXTENSION_PATHS]


def test_immutable_v4_without_closure_metadata_keeps_exact_subtree_verification():
    base, extensions = chain()
    assert "subtree_closure" not in extensions[3]["authority"]
    seal._verify_chain(base, extensions)
    assert seal.verify()["status"] == "PASS__NATIVE_RUNTIME_SOURCE_SEAL_CHAIN"


@pytest.mark.parametrize("version", [3, 5, 16])
def test_later_closure_claim_cannot_be_omitted(version):
    base, extensions = chain()
    del extensions[version - 1]["authority"]["subtree_closure"]
    with pytest.raises(RuntimeError, match=f"EXT{version}_SUBTREE_CLOSURE_DRIFT"):
        seal._verify_chain(base, extensions)


def test_v4_exception_does_not_accept_an_incorrect_claim_or_skip_an_extension():
    base, extensions = chain()
    wrong = deepcopy(extensions)
    wrong[3]["authority"]["subtree_closure"] = "PARTIAL"
    with pytest.raises(RuntimeError, match="EXT4_SUBTREE_CLOSURE_DRIFT"):
        seal._verify_chain(base, wrong)
    with pytest.raises(RuntimeError, match="CHAIN_LENGTH_DRIFT"):
        seal._verify_chain(base, extensions[:-1])


def test_historical_bytes_and_unsealed_native_files_still_fail(monkeypatch):
    base, extensions = chain()
    wrong = deepcopy(extensions)
    wrong[4]["prior_extension"]["git_blob_sha1"] = "0" * 40
    with pytest.raises(RuntimeError, match="EXT5_PRIOR_BLOB_DRIFT"):
        seal._verify_chain(base, wrong)
    original = seal.subprocess.check_output

    def with_unsealed_file(args, **kwargs):
        if args[:2] == ["git", "ls-files"]:
            return original(args, **kwargs) + "runtime/realsas_cpp/unsealed.cpp\n"
        return original(args, **kwargs)

    monkeypatch.setattr(seal.subprocess, "check_output", with_unsealed_file)
    with pytest.raises(RuntimeError, match="SUBTREE.*DRIFT|UNSEALED"):
        seal.verify()
