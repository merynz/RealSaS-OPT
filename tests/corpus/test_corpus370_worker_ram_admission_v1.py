import pytest
from tools.notebooks.build_corpus370_compute_worker_v1 import effective_memory_gib


@pytest.mark.parametrize('host,limits,expected',[
    (256,['8589934592'],8),
    (256,['max','34359738368'],32),
    (24,['max','9223372036854771712'],24),
    (16,[],16),
    (32,['invalid','0','-1'],32),
])
def test_capacity_uses_the_smallest_actual_host_or_container_limit(host,limits,expected):
    assert effective_memory_gib(f'MemTotal: {host*1024**2} kB\n',limits)==expected
