"""Fixed-source 32-way quarantine partitions never overlap."""
import hashlib
from collections import Counter

from tools.corpus.recover_corpus4008_github_source_shard_v1 import owner_for_sha


def test_all_sha_pinned_candidates_assigned_once():
    keys=[hashlib.sha256(('synthetic'+str(i)).encode()).hexdigest() for i in range(3349)]
    ownership={k:owner_for_sha(k,32) for k in keys}
    assert len(ownership)==3349
    assert set(ownership.values())==set(range(32))
    assert sum(Counter(ownership.values()).values())==3349
    assert all(0<=value<32 for value in ownership.values())
    assert all(owner_for_sha(k,32)==ownership[k] for k in keys)


def test_shard_distribution_has_no_empty_partition_for_diverse_3349():
    keys=[hashlib.sha256(('synthetic'+str(i)).encode()).hexdigest() for i in range(3349)]
    counts=Counter(owner_for_sha(k,32) for k in keys)
    assert len(counts)==32 and min(counts.values())>0
