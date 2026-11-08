"""Audited loader for the unmodified upstream THItoGene source tree."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, ModuleType

import torch
import torch.nn.functional as F


EXPECTED_REVISION = "503e37ac9cf994f0cb14e02a608c45a204ebdfd1"
ATTENTION_BACKEND = "torch_flash_sdpa_bfloat16_and_edgewise_exact_gat_full_section"
EXPECTED_PACKAGES = {
    "torch": "1.13.1",
    "torchvision": "0.14.1",
    "pytorch-lightning": "2.0.6",
    "torchmetrics": "1.0.1",
    "einops": "0.6.1",
    "scprep": "1.2.3",
}


class UpstreamContractError(RuntimeError):
    """Raised rather than substituting a different model or environment."""


@dataclass(frozen=True)
class UpstreamReceipt:
    root: str
    revision: str
    expected_revision: str
    source_files: tuple[str, ...]
    package_versions: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "revision": self.revision,
            "expected_revision": self.expected_revision,
            "source_files": list(self.source_files),
            "package_versions": dict(self.package_versions),
        }


def default_upstream_root() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "tmp"
        / "method_repos"
        / "sCellST"
        / "external"
        / "THItoGene"
    )


def _revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise UpstreamContractError(
            f"UPSTREAM_REVISION_UNAVAILABLE:{root}:{result.stderr.strip()}"
        )
    return result.stdout.strip()


def verify_upstream_source(
    root: Path | None = None, *, require_expected_revision: bool = True
) -> UpstreamReceipt:
    resolved = (root or default_upstream_root()).resolve()
    required = (
        "vis_model.py",
        "ODConv.py",
        "efficient_capsnet.py",
        "transformer.py",
        "GATLayer.py",
        "graph_construction.py",
        "requirements.txt",
        "THItoGene.yml",
    )
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise UpstreamContractError(f"UPSTREAM_SOURCE_MISSING:{resolved}:{missing}")
    revision = _revision(resolved)
    if require_expected_revision and revision != EXPECTED_REVISION:
        raise UpstreamContractError(
            f"UPSTREAM_REVISION_DRIFT:{revision}:{EXPECTED_REVISION}"
        )
    versions: dict[str, str] = {}
    for package in EXPECTED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    return UpstreamReceipt(
        root=str(resolved),
        revision=revision,
        expected_revision=EXPECTED_REVISION,
        source_files=required,
        package_versions=versions,
    )


def assert_pinned_environment() -> dict[str, str]:
    observed: dict[str, str] = {}
    mismatches: list[str] = []
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            value = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            value = "MISSING"
        observed[package] = value
        # Local version suffixes such as +cu117 are allowed for the pinned
        # torch build; the public version must still begin with the frozen one.
        if not value.startswith(expected):
            mismatches.append(f"{package}={value},expected={expected}")
    if sys.version_info[:2] != (3, 9):
        mismatches.append(
            f"python={sys.version_info.major}.{sys.version_info.minor},expected=3.9"
        )
    if mismatches:
        raise UpstreamContractError(
            "UPSTREAM_ENVIRONMENT_MISMATCH:" + ";".join(mismatches)
        )
    return observed


def _load_source_module(root: Path, filename: str, module_name: str) -> ModuleType:
    source = root / filename
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise UpstreamContractError(f"UPSTREAM_IMPORT_SPEC:{source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_upstream_modules(
    root: Path | None = None,
    *,
    require_pinned_environment: bool = True,
) -> tuple[ModuleType, ModuleType, UpstreamReceipt]:
    receipt = verify_upstream_source(root)
    resolved = Path(receipt.root)
    if require_pinned_environment:
        assert_pinned_environment()
    # Upstream uses absolute intra-repository imports.  Inserting its audited
    # root is necessary; the source itself remains untouched.
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    vis_model = _load_source_module(
        resolved, "vis_model.py", "g0055_upstream_thitogene_vis_model"
    )
    graph = _load_source_module(
        resolved, "graph_construction.py", "g0055_upstream_thitogene_graph"
    )
    if not hasattr(vis_model, "THItoGene") or not hasattr(graph, "calcADJ"):
        raise UpstreamContractError("UPSTREAM_API_MISSING")
    return vis_model, graph, receipt


def _memory_efficient_attention(self, x):
    """The upstream attention equation through PyTorch's streaming SDPA kernel."""

    batch, spots, _ = x.shape
    qkv = self.to_qkv(x).chunk(3, dim=-1)
    q, k, v = (
        value.reshape(batch, spots, self.heads, -1).transpose(1, 2)
        for value in qkv
    )
    if x.is_cuda:
        # THItoGene's explicit fp32 [B,H,N,N] attention is the source of the
        # section-size OOM.  Flash SDPA evaluates the identical full-section
        # equation without materialising N squared weights.  Only Q/K/V and
        # the attention result use bfloat16; the surrounding published model
        # remains fp32.
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q, k, v = (value.to(torch.bfloat16) for value in (q, k, v))
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                scale=float(self.scale),
            )
        out = out.to(dtype=x.dtype)
    else:
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            scale=float(self.scale),
        )
    out = out.transpose(1, 2).contiguous().reshape(batch, spots, -1)
    return self.to_out(out)


def _edge_index(adjacency: torch.Tensor, *, nodes: int) -> torch.Tensor:
    if (
        adjacency.ndim == 2
        and adjacency.shape[0] == 2
        and adjacency.dtype in (torch.int32, torch.int64)
    ):
        edges = adjacency.to(dtype=torch.long)
    else:
        dense = adjacency[0] if adjacency.ndim == 3 and adjacency.shape[0] == 1 else adjacency
        if dense.shape != (nodes, nodes):
            raise UpstreamContractError(
                f"UPSTREAM_GAT_ADJACENCY_SHAPE:{tuple(adjacency.shape)}:{nodes}"
            )
        edges = torch.nonzero(dense > 0, as_tuple=False).transpose(0, 1).contiguous()
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.numel() == 0:
        raise UpstreamContractError(f"UPSTREAM_GAT_EDGE_INDEX:{tuple(edges.shape)}")
    return edges


def _memory_efficient_graph_attention(self, h, adjacency):
    """Evaluate the upstream dense GAT equation over its frozen non-zero edges."""

    nodes = h.shape[0]
    edges = _edge_index(adjacency, nodes=nodes)
    src, dst = edges[0], edges[1]
    wh = torch.mm(h, self.W)
    left = torch.matmul(wh, self.a[: self.out_features, :]).squeeze(-1)
    right = torch.matmul(wh, self.a[self.out_features :, :]).squeeze(-1)
    scores = self.leakyrelu(left[src] + right[dst])

    maximum = torch.full(
        (nodes,), -torch.inf, dtype=scores.dtype, device=scores.device
    )
    maximum.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
    weights = torch.exp(scores - maximum[dst])
    denominator = torch.zeros(nodes, dtype=scores.dtype, device=scores.device)
    denominator.scatter_add_(0, dst, weights)
    weights = weights / denominator[dst]
    weights = F.dropout(weights, self.dropout, training=self.training)

    output = torch.zeros(
        (nodes, self.out_features), dtype=wh.dtype, device=wh.device
    )
    output.index_add_(0, src, weights.unsqueeze(-1) * wh[dst])

    # With the upstream [1,N,N] adjacency, softmax(dim=1) assigns a uniform
    # 1/N column when a destination has no incoming 4-NN edge.  Preserve that
    # unusual but published behavior without recreating an N-by-N tensor.
    empty = torch.nonzero(denominator == 0, as_tuple=False).squeeze(-1)
    for start in range(0, len(empty), 256):
        empty_chunk = empty[start : start + 256]
        uniform = torch.full(
            (nodes, len(empty_chunk)),
            1.0 / nodes,
            dtype=wh.dtype,
            device=wh.device,
        )
        uniform = F.dropout(uniform, self.dropout, training=self.training)
        output = output + torch.matmul(uniform, wh[empty_chunk])

    result = F.elu(output) if self.concat else output
    return result.unsqueeze(0)


def instantiate_upstream_model(
    vis_model: ModuleType,
    *,
    n_genes: int = 50,
):
    """Instantiate the published HER2ST architecture with only output width changed."""

    model = vis_model.THItoGene(
        patch_size=112,
        n_layers=4,
        n_genes=n_genes,
        dim=1024,
        learning_rate=1e-5,
        dropout=0.2,
        n_pos=64,
        heads=[16, 8],
        caps=20,
        route_dim=64,
    )
    routing = model.caps_layer.digit_caps
    routing_module = sys.modules[type(routing).__module__]
    squash = routing_module.squash

    def forward_device_compatible(self, input):
        u = torch.einsum("...ji,kjiz->...kjz", input, self.W)
        c = torch.einsum("...ij,...kj->...i", u, u)[..., None]
        c = c / math.sqrt(self.dim_capsules)
        c = torch.softmax(c, axis=1)
        c = c + self.b
        s = torch.sum(torch.mul(u, c), dim=-2)
        return squash(s)

    routing.forward = MethodType(forward_device_compatible, routing)

    attention_layers = 0
    graph_attention_layers = 0
    for module in model.modules():
        if (
            type(module).__name__ == "Attention"
            and hasattr(module, "to_qkv")
            and hasattr(module, "to_out")
            and hasattr(module, "heads")
            and hasattr(module, "scale")
        ):
            module.forward = MethodType(_memory_efficient_attention, module)
            attention_layers += 1
        if (
            type(module).__name__ == "GraphAttentionLayer"
            and hasattr(module, "W")
            and hasattr(module, "a")
            and hasattr(module, "out_features")
            and hasattr(module, "leakyrelu")
        ):
            module.forward = MethodType(_memory_efficient_graph_attention, module)
            graph_attention_layers += 1
    if attention_layers < 1:
        raise UpstreamContractError("UPSTREAM_ATTENTION_LAYER_MISSING")
    if graph_attention_layers < 1:
        raise UpstreamContractError("UPSTREAM_GRAPH_ATTENTION_LAYER_MISSING")
    model._g0055_attention_backend = ATTENTION_BACKEND
    model._g0055_graph_attention_layers = graph_attention_layers
    return model
