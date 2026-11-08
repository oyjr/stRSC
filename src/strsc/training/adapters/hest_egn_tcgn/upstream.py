"""Audited loaders for the pinned EGN and TCGN publication repositories."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator, Optional, Union

import torch
from torch import nn

from .contracts import MODEL_CONFIGS, canonical_model_name


class UpstreamContractError(RuntimeError):
    """Raised rather than using a substitute architecture or dependency set."""


EXPECTED_PACKAGES = {
    "EGNv1": {
        "torch": "1.9.1",
        "torchvision": "0.10.1",
        "pytorch-lightning": "1.5.7",
        "numpy": "1.23.3",
        "pandas": "1.5.2",
        "einops": "0.3.2",
        "tifffile": "2021.7.2",
        "Pillow": "9.3.0",
        "scanpy": "1.9.1",
    },
    "EGNv2": {
        "torch": "1.9.1",
        "torchvision": "0.10.1",
        "pytorch-lightning": "1.5.7",
        "numpy": "1.23.3",
        "pandas": "1.5.2",
        "torch-geometric": "2.1.0.post1",
        "tifffile": "2021.7.2",
        "Pillow": "9.3.0",
        "scanpy": "1.9.1",
    },
    # The publication only states lower bounds.  The fixed attempt records the
    # observed versions and enforces the documented Python/PyTorch minima.
    "TCGN": {
        "torch": ">=1.10",
        "scanpy": ">=1.8",
        "torchvision": "PRESENT",
        "scipy": "PRESENT",
        "einops": "PRESENT",
        "scprep": "PRESENT",
        "timm": "PRESENT",
        "memory-profiler": "PRESENT",
    },
}


@dataclass(frozen=True)
class UpstreamReceipt:
    model: str
    root: str
    revision: str
    expected_revision: str
    source_files: tuple[str, ...]
    package_versions: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "root": self.root,
            "revision": self.revision,
            "expected_revision": self.expected_revision,
            "source_files": list(self.source_files),
            "package_versions": dict(self.package_versions),
        }


def default_repository_root(model: str) -> Path:
    name = canonical_model_name(model)
    return Path(__file__).resolve().parents[2] / "deps" / MODEL_CONFIGS[name].repository


def _git_revision(root: Path) -> str:
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


def _required_files(model: str) -> tuple[str, ...]:
    if model == "EGNv1":
        return ("README.md", "v1/egn.py", "v1/train.py", "v1/dataset.py", "v1/build_exemplar.py")
    if model == "EGNv2":
        return (
            "README.md",
            "v2/model.py",
            "v2/train.py",
            "v2/generate_graph.py",
            "v2/build_exemplar.py",
            "v2/geb.py",
            "v2/heteroconv.py",
            "v2/csra.py",
        )
    return (
        "readme.md",
        "model.py",
        "train.py",
        "dataloader.py",
        "dataset.py",
        "CMT_block.py",
        "gnn_block.py",
        "transformer_block.py",
        "censnet_block.py",
    )


def _package_versions(model: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for package in EXPECTED_PACKAGES[model]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "MISSING"
    return result


def verify_upstream_source(
    model: str,
    root: Optional[Path] = None,
    *,
    require_expected_revision: bool = True,
) -> UpstreamReceipt:
    name = canonical_model_name(model)
    config = MODEL_CONFIGS[name]
    resolved = (root or default_repository_root(name)).resolve()
    required = _required_files(name)
    missing = [path for path in required if not (resolved / path).is_file()]
    if missing:
        raise UpstreamContractError(f"UPSTREAM_SOURCE_MISSING:{resolved}:{missing}")
    revision = _git_revision(resolved)
    if require_expected_revision and revision != config.revision:
        raise UpstreamContractError(
            f"UPSTREAM_REVISION_DRIFT:{name}:{revision}:{config.revision}"
        )
    return UpstreamReceipt(
        model=name,
        root=str(resolved),
        revision=revision,
        expected_revision=config.revision,
        source_files=required,
        package_versions=_package_versions(name),
    )


def assert_pinned_environment(model: str) -> dict[str, str]:
    name = canonical_model_name(model)
    observed = _package_versions(name)
    mismatches: list[str] = []
    if name.startswith("EGN"):
        for package, expected in EXPECTED_PACKAGES[name].items():
            value = observed[package]
            if not value.startswith(expected):
                mismatches.append(f"{package}={value},expected={expected}")
        if sys.version_info[:2] != (3, 9):
            mismatches.append(
                f"python={sys.version_info.major}.{sys.version_info.minor},expected=3.9"
            )
    else:
        if observed["torch"] == "MISSING":
            mismatches.append("torch=MISSING,expected>=1.10")
        else:
            parts = observed["torch"].split("+")[0].split(".")[:2]
            try:
                if tuple(map(int, parts)) < (1, 10):
                    mismatches.append(f"torch={observed['torch']},expected>=1.10")
            except ValueError:
                mismatches.append(f"torch_version_unparseable={observed['torch']}")
        if observed["scanpy"] == "MISSING":
            mismatches.append("scanpy=MISSING,expected>=1.8")
        else:
            try:
                scanpy_parts = tuple(
                    int(part) for part in observed["scanpy"].split(".")[:2]
                )
                if scanpy_parts < (1, 8):
                    mismatches.append(
                        f"scanpy={observed['scanpy']},expected>=1.8"
                    )
            except ValueError:
                mismatches.append(
                    f"scanpy_version_unparseable={observed['scanpy']}"
                )
        for package, expected in EXPECTED_PACKAGES[name].items():
            if expected == "PRESENT" and observed[package] == "MISSING":
                mismatches.append(f"{package}=MISSING")
        if sys.version_info < (3, 7):
            mismatches.append(
                f"python={sys.version_info.major}.{sys.version_info.minor},expected>=3.7"
            )
    if mismatches:
        raise UpstreamContractError(
            "UPSTREAM_ENVIRONMENT_MISMATCH:" + ";".join(mismatches)
        )
    return observed


@contextmanager
def _source_path(path: Path) -> Iterator[None]:
    value = str(path)
    already = value in sys.path
    if not already:
        sys.path.insert(0, value)
    try:
        yield
    finally:
        if not already and value in sys.path:
            sys.path.remove(value)


def _load_file(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise UpstreamContractError(f"UPSTREAM_IMPORT_SPEC:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_upstream_model_module(
    model: str,
    root: Optional[Path] = None,
    *,
    require_pinned_environment: bool = True,
) -> tuple[ModuleType, UpstreamReceipt]:
    name = canonical_model_name(model)
    receipt = verify_upstream_source(name, root)
    if require_pinned_environment:
        assert_pinned_environment(name)
    repository = Path(receipt.root)
    source_dir = repository / MODEL_CONFIGS[name].source_subdir
    filename = "egn.py" if name == "EGNv1" else "model.py"
    with _source_path(source_dir):
        module = _load_file(
            source_dir / filename,
            f"g0055_upstream_{name.lower()}_{abs(hash(str(source_dir)))}",
        )
    expected = "EGN" if name == "EGNv1" else ("HeteroGNN" if name == "EGNv2" else "TCGN")
    if not hasattr(module, expected):
        raise UpstreamContractError(f"UPSTREAM_API_MISSING:{name}:{expected}")
    return module, receipt


class EGNv2GeneWidthAdapter(nn.Module):
    """The published EGNv2 forward with its hard-coded 250 changed to 50.

    All graph layers, aggregation, pooling and expression-conditioned exemplar
    inputs are the original modules.  This wrapper exists only because the
    publication source hard-codes the gene width in both layers and slicing.
    """

    def __init__(self, upstream: nn.Module, n_genes: int):
        super().__init__()
        self.upstream = upstream
        self.n_genes = int(n_genes)

    def forward(self, x_dict, edge_index_dict):
        # Avoid the publication forward's in-place mutation of caller state.
        x = dict(x_dict)
        model = self.upstream
        x["example"] = model.post_transform(model.pretransform_exp(x["example"]))
        x["window"] = model.post_transform(model.pretransform_win(x["window"]))
        x["example_y"] = model.pretransform_ey(x["example"][:, -self.n_genes :])
        for conv in model.convs:
            x = conv(x, edge_index_dict)
            x = {key: model.leaklyrelu(value) for key, value in x.items()}
        return model.lin(model.pool(x, edge_index_dict))


def instantiate_upstream_model(
    model: str,
    module: ModuleType,
    *,
    n_genes: int = 50,
) -> nn.Module:
    name = canonical_model_name(model)
    if n_genes != 50:
        raise UpstreamContractError(f"G0055_GENE_WIDTH:{n_genes}")
    if name == "TCGN":
        return module.TCGN(num_classes=n_genes)
    if name == "EGNv1":
        result = module.EGN(
            image_size=256,
            dim=1024,
            depth=8,
            heads=16,
            mlp_dim=4096,
            bhead=8,
            bdim=64,
            bfre=2,
            mdim=2048,
            player=2,
            linear_projection=True,
            num_classes=n_genes,
        )
        projector = result.transformer.encoder
        projector.proe = nn.Linear(2048 + n_genes, 1024)
        projector.to_v = nn.Linear(n_genes, 1024)
        return result

    result = module.HeteroGNN(num_layers=4, mdim=512)
    pyg = module.pyg
    result.pretransform_exp = pyg.Linear(512 + n_genes, 512, bias=False)
    result.pretransform_ey = pyg.Linear(n_genes, 512, bias=False)
    result.lin = pyg.Linear(512, n_genes)
    return EGNv2GeneWidthAdapter(result, n_genes)


def load_tcgn_pretraining(
    model: nn.Module, *, path: Path, map_location: Union[torch.device, str]
) -> dict[str, list[str]]:
    """Apply the publication's ``cmt_tiny.pth, strict=False`` initialization."""

    if not path.is_file():
        raise UpstreamContractError(f"TCGN_PUBLISHED_PRETRAINING_MISSING:{path}")
    state = torch.load(path, map_location=map_location)
    try:
        incompatible = model.load_state_dict(state, strict=False)
    except RuntimeError as error:
        raise UpstreamContractError(f"TCGN_PRETRAINING_INCOMPATIBLE:{error}") from error
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
