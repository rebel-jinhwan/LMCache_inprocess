# SPDX-License-Identifier: Apache-2.0
"""LMCache ignores the handshake metadata of every pipeline stage.

It takes no part in cross-instance transfers, so a ``pp_rank > 0`` entry,
which vLLM's default rejects, must not fail a ``MultiConnector`` that holds
it beside a PP-aware transfer connector.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorBase_V1,
)

# First Party
from lmcache.inprocess.integration.vllm.lmcache_connector_v1 import (  # noqa: E402
    LMCacheConnectorV1Dynamic,
)


def _metadata(pp_size: int, tp_size: int) -> dict:
    return {
        (pp_rank, tp_rank): object()
        for pp_rank in range(pp_size)
        for tp_rank in range(tp_size)
    }


def test_vllm_default_rejects_pp_handshake_metadata() -> None:
    # Guards the premise: if vLLM stops raising, the override is moot.
    hook = "set_xfer_handshake_metadata_pp_aware"
    if hook not in vars(KVConnectorBase_V1):  # pragma: no cover - older vLLM
        pytest.skip("vLLM has no PP-aware handshake hook")
    with pytest.raises(ValueError, match="pp_rank > 0"):
        KVConnectorBase_V1.set_xfer_handshake_metadata_pp_aware(
            SimpleNamespace(), _metadata(pp_size=2, tp_size=1)
        )


@pytest.mark.parametrize("pp_size,tp_size", [(1, 1), (2, 1), (4, 2)])
def test_connector_ignores_handshake_metadata_of_every_stage(
    pp_size: int, tp_size: int
) -> None:
    # The hook touches no connector state, so skip the engine-building init.
    connector = object.__new__(LMCacheConnectorV1Dynamic)
    assert (
        connector.set_xfer_handshake_metadata_pp_aware(_metadata(pp_size, tp_size))
        is None
    )
