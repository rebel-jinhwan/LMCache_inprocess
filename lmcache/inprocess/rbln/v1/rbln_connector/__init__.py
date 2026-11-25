# SPDX-License-Identifier: Apache-2.0
"""RBLN connector public surface.

:class:`RBLNConnector` is the single KV connector variant: KV caches are
real ``torch`` tensors on ``device='rbln'`` (vllm-rbln runs in device-tensor
mode) and KV transfer uses torch device ops only.
"""

# First Party
from lmcache.inprocess.v1.gpu_connector.gpu_connectors import GPUConnectorInterface

# Local
from .connector import RBLNConnector, sized_head_groups

__all__ = [
    "GPUConnectorInterface",
    "RBLNConnector",
    "sized_head_groups",
]
