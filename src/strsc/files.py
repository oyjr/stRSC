"""Read RNA counts and predictions for :func:`strsc.rsc`.

Counts can come as

* a CSV or TSV table: ``sample_id, unit_id, x, y``, optional ``patient_id``, ``region``,
  ``cell_class`` and ``area`` columns, then one column per gene (the format of the website);
* an AnnData ``.h5ad`` file, for example a HEST-1k sample;
* a 10x Genomics feature-barcode matrix with its positions: an ``.h5`` file or a
  ``filtered_feature_bc_matrix`` folder (``matrix.mtx.gz``, ``features.tsv.gz``, ``barcodes.tsv.gz``),
  with the ``tissue_positions`` file of Space Ranger (Visium spots or Visium HD bins), or the
  ``cells`` file of Xenium;
* a folder holding one of these, such as a Space Ranger ``outs`` folder, a Visium HD
  ``square_016um`` folder or a Xenium output folder. A folder that holds several such folders
  gives one section per folder.

Predictions are CSV or TSV tables with ``sample_id`` (optional when there is one section),
``unit_id`` and one column per gene, or ``.h5ad`` files whose observation names are the unit IDs.

HDF5 files (``.h5ad``, ``.h5``) need h5py, and Parquet files need pyarrow.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

ID_COLUMNS = ("sample_id", "unit_id")
METADATA = ("sample_id", "unit_id", "patient_id", "x", "y", "region", "cell_class", "area")
TEXT_COLUMNS = ("sample_id", "unit_id", "patient_id", "region", "cell_class")
MISSING = ["", "NA", "na", "NaN", "nan", "N/A", "n/a", "null", "NULL", "None", "none"]
# Xenium probes that are not genes (the paper's cell protocol drops the same prefixes).
CONTROL_PREFIXES = ("BLANK_", "NegControlCodeword_", "NegControlProbe_", "antisense_",
                    "UnassignedCodeword_", "DeprecatedCodeword_", "Intergenic_Region")
VISIUM_SPOT_UM = 55.0  # diameter of a Visium spot, which scalefactors_json.json gives in full-resolution pixels
TABLE_ENDINGS = (".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz")


class InputError(ValueError):
    """A problem with the input files that the user can fix."""


@dataclass
class Counts:
    """Molecule counts of one or more sections, with the position of every spot, bin or cell."""

    counts: sparse.csr_matrix                 # locations x genes, nonnegative integers
    genes: list[str]
    ids: np.ndarray                           # unit ID of each location
    sections: np.ndarray                      # section of each location
    xy: np.ndarray                            # (locations, 2) coordinates
    unit: str = "input"                       # "um" when the coordinates are micrometers
    columns: pd.DataFrame = field(default_factory=pd.DataFrame)  # region, cell_class, patient_id, area, ...
    spacing_um: float | None = None           # grid spacing, when the file says it (Visium HD bins)
    source: str = ""

    def __post_init__(self) -> None:
        n = self.counts.shape[0]
        if len(self.ids) != n or len(self.sections) != n or self.xy.shape != (n, 2):
            raise InputError("The counts, IDs, sections and coordinates do not have the same number of rows.")
        if self.columns is None or self.columns.empty:
            self.columns = pd.DataFrame(index=pd.RangeIndex(n))
        self.columns = self.columns.reset_index(drop=True)

    def __len__(self) -> int:
        return int(self.counts.shape[0])

    def __repr__(self) -> str:
        names = self.section_names()
        shown = ", ".join(names[:4]) + (", ..." if len(names) > 4 else "")
        extra = [c for c in self.columns.columns]
        return (f"Counts({len(self):,} locations, {len(self.genes):,} genes, {len(names)} section"
                f"{'s' if len(names) != 1 else ''}: {shown}; coordinates in {'µm' if self.unit == 'um' else 'input units'}"
                + (f"; columns {', '.join(extra)}" if extra else "") + ")")

    def section_names(self) -> list[str]:
        return [str(s) for s in pd.unique(self.sections)]

    def rows(self, section: str) -> np.ndarray:
        return np.flatnonzero(self.sections == section)

    def column(self, name: str) -> np.ndarray | None:
        return self.columns[name].to_numpy() if name in self.columns.columns else None

    def keep(self, rows: np.ndarray) -> "Counts":
        rows = np.asarray(rows)
        return Counts(self.counts[rows].tocsr(), list(self.genes), self.ids[rows], self.sections[rows], self.xy[rows],
                      self.unit, self.columns.iloc[rows].reset_index(drop=True), self.spacing_um, self.source)

    def with_labels(self, path: str | Path, name: str | None = None) -> "Counts":
        """Add columns such as ``region`` or ``cell_class`` from a CSV or TSV table with ``unit_id``
        (and ``sample_id`` when there are several sections)."""
        frame = _read_plain_table(Path(path), name or Path(path).name)
        index = _location_index(frame, self, name or Path(path).name)
        out = self.columns.copy()
        for column in frame.columns:
            if column in ("sample_id", "unit_id"):
                continue
            values = pd.Series(frame[column].to_numpy(), index=index)
            out[column] = values.reindex(range(len(self))).to_numpy()
        return Counts(self.counts, self.genes, self.ids, self.sections, self.xy, self.unit, out, self.spacing_um, self.source)


@dataclass
class Prediction:
    """Predicted expression of one model on the locations and genes of a :class:`Counts`."""

    name: str
    values: np.ndarray          # locations x genes of the counts, NaN where not predicted
    genes: list[str]            # genes the file predicts, in the order of the counts
    source: str = ""

    def __repr__(self) -> str:
        covered = int(np.isfinite(self.values).all(axis=1).sum()) if self.values.size else 0
        return f"Prediction({self.name!r}: {len(self.genes):,} genes, {covered:,} locations with every gene)"


# ---------------------------------------------------------------- counts

def read_counts(
    path: str | Path | Sequence[str | Path],
    *,
    sample: str | None = None,
    positions: str | Path | None = None,
    layer: str | None = None,
    spatial_key: str = "spatial",
    sample_column: str | None = None,
    columns: Iterable[str] | None = None,
    um_per_unit: float | None = None,
) -> Counts:
    """Read molecule counts.

    ``path`` is a table, an ``.h5ad`` file, a 10x ``.h5`` matrix, a folder, or a list of these
    (one section each). ``sample`` names the section when the file does not; otherwise the file
    or folder name is used. ``positions`` gives the positions file of a 10x matrix when it is not
    next to the matrix. For ``.h5ad`` files, ``layer`` picks a layer of raw counts (by default
    ``layers['counts']`` when present, else ``X``), ``spatial_key`` the coordinates in ``obsm``,
    ``sample_column`` an ``obs`` column that names the sections, and ``columns`` further ``obs``
    columns to keep, such as regions or cell classes. ``um_per_unit`` converts coordinates to
    micrometers when the file does not say how.
    """
    if isinstance(path, (list, tuple)):
        parts = [read_counts(p, positions=positions, layer=layer, spatial_key=spatial_key, sample_column=sample_column,
                             columns=columns, um_per_unit=um_per_unit) for p in path]
        return concat(parts)
    path = Path(path)
    if not path.exists():
        raise InputError(f"{path} does not exist.")
    if path.is_dir():
        return _read_folder(path, sample, layer, spatial_key, sample_column, columns, um_per_unit)
    name = path.name.lower()
    if name in _MTX_MATRIX:
        return _read_folder(path.parent, sample, layer, spatial_key, sample_column, columns, um_per_unit)
    if name.endswith(".h5ad"):
        return _read_h5ad(path, sample, layer, spatial_key, sample_column, columns, um_per_unit)
    if name.endswith(".h5"):
        found = Path(positions) if positions else _positions_next_to(path)
        if found is None:
            raise InputError(f"{path.name}: also give the positions of its barcodes, the tissue_positions file of Space Ranger or the cells file of Xenium.")
        return _read_10x(path, found, sample or _section_name(path), um_per_unit)
    if name.endswith(TABLE_ENDINGS):
        return _read_counts_table(path, path.name, sample, um_per_unit)
    raise InputError(f"{path.name}: not a format strsc reads. Use a CSV or TSV table, an .h5ad file, or a 10x .h5 matrix with its positions.")


def read_files(files: Sequence[tuple[str | Path, str]], **options) -> Counts:
    """Counts from files picked one by one, each given as (path on disk, original name with its folder).
    A 10x matrix finds its positions and scale factors in its own folder or in the ``spatial`` folder
    next to it, as Space Ranger writes them; every other table or .h5ad file is a section of its own."""
    items = [(Path(disk), str(original).replace("\\", "/")) for disk, original in files]
    folder = lambda original: str(Path(original).parent).replace("\\", "/")
    name = lambda original: Path(original).name.lower()
    matrices = [(d, o) for d, o in items if name(o).endswith(".h5") and not name(o).endswith(".h5ad")]
    used = set(matrices)
    by_folder: dict[str, list[tuple[Path, str]]] = {}
    for d, o in matrices:
        by_folder.setdefault(folder(o), []).append((d, o))
    parts = []
    for place, group in by_folder.items():
        if len(group) > 1:
            raise InputError(f"{place if place != '.' else 'The chosen files'}: more than one .h5 matrix in one folder. Put each section in its own folder.")
        disk, original = group[0]
        near = [(d, o) for d, o in items if folder(o) in (place, (place + "/spatial") if place != "." else "spatial")]
        position = next(((d, o) for wanted in _POSITION_NAMES for d, o in near if name(o) == wanted), None)
        if position is None:
            raise InputError(f"{Path(original).name}: choose its positions too, the tissue_positions file of Space Ranger or the cells file of Xenium.")
        scalefactors = next(((d, o) for d, o in near if name(o) == "scalefactors_json.json"), None)
        used.update([position] + ([scalefactors] if scalefactors else []))
        section = options.get("sample") or _section_name(Path(original))
        parts.append(_read_10x(disk, position[0], section, options.get("um_per_unit"), scalefactors[0] if scalefactors else None,
                               original_name=Path(original).name))
    mtx = [(d, o) for d, o in items if name(o) in _MTX_MATRIX]
    for disk, original in mtx:
        place = folder(original)
        here = {name(o): (d, o) for d, o in items if folder(o) == place}
        features = next((here[n] for n in _MTX_FEATURES if n in here), None)
        barcodes = next((here[n] for n in _MTX_BARCODES if n in here), None)
        if features is None or barcodes is None:
            raise InputError(f"{original}: choose the whole folder, with features.tsv.gz and barcodes.tsv.gz next to the matrix.")
        parent = str(Path(place).parent).replace("\\", "/") if place != "." else "."
        near = [(d, o) for d, o in items if folder(o) in {place, place + "/spatial", parent + "/spatial" if parent != "." else "spatial"}]
        position = next(((d, o) for wanted in _POSITION_NAMES for d, o in near if name(o) == wanted), None)
        if position is None:
            raise InputError(f"{original}: choose its positions too, the tissue_positions file in the spatial folder of Space Ranger.")
        scalefactors = next(((d, o) for d, o in near if name(o) == "scalefactors_json.json"), None)
        used.update([(disk, original), features, barcodes, position] + ([scalefactors] if scalefactors else []))
        section = options.get("sample") or _section_name(Path(original))
        parts.append(_read_10x_mtx({"matrix": disk, "features": features[0], "barcodes": barcodes[0]}, position[0], section,
                                   options.get("um_per_unit"), scalefactors[0] if scalefactors else None, original_name=original))
    for disk, original in items:
        if (disk, original) in used:
            continue
        lower = name(original)
        if lower.endswith(".h5ad"):
            parts.append(_read_h5ad(disk, options.get("sample") or _section_name(Path(original)), options.get("layer"),
                                    options.get("spatial_key", "spatial"), options.get("sample_column"), options.get("columns"),
                                    options.get("um_per_unit"), original_name=Path(original).name))
        elif lower.endswith(TABLE_ENDINGS) and lower not in _POSITION_NAMES and lower not in _MTX_NAMES:
            parts.append(_read_counts_table(disk, Path(original).name, options.get("sample"), options.get("um_per_unit")))
    if not parts:
        raise InputError("None of the chosen files holds counts: a CSV or TSV table, an .h5ad file, or a 10x .h5 matrix or filtered_feature_bc_matrix folder with its positions.")
    return concat(parts)


def concat(parts: Sequence[Counts]) -> Counts:
    """Join the counts of several sections. Genes are those of the first section; every section needs them."""
    if len(parts) == 1:
        return parts[0]
    genes = parts[0].genes
    for part in parts[1:]:
        missing = [g for g in genes if g not in set(part.genes)]
        if missing:
            raise InputError(f"Section {part.section_names()[0]} lacks {len(missing):,} of the genes of {parts[0].section_names()[0]}, for example {missing[0]}. Every section needs the same genes.")
    names = [n for part in parts for n in part.section_names()]
    if len(set(names)) != len(names):
        raise InputError("Two sections have the same name. Rename the files or folders.")
    units = {part.unit for part in parts}
    blocks = []
    for part in parts:
        order = [part.genes.index(g) for g in genes]
        blocks.append(part.counts[:, order])
    return Counts(sparse.vstack(blocks).tocsr(), list(genes), np.concatenate([p.ids for p in parts]),
                  np.concatenate([p.sections for p in parts]), np.vstack([p.xy for p in parts]),
                  units.pop() if len(units) == 1 else "input", pd.concat([p.columns for p in parts], ignore_index=True),
                  parts[0].spacing_um if len({p.spacing_um for p in parts}) == 1 else None, "; ".join(p.source for p in parts))


_GENERIC = {"outs", "spatial", "binned_outputs", "filtered_feature_bc_matrix", "raw_feature_bc_matrix", "cell_feature_matrix", "matrix", ""}


def _section_name(path: Path) -> str:
    """A section's name from its file or folder. Generic 10x names (outs, square_016um, the matrix
    file itself) are skipped in favor of the folder above them."""
    stem = path.name
    for ending in (".h5ad", ".h5", ".mtx.gz", ".mtx", ".csv.gz", ".tsv.gz", ".txt.gz", ".csv", ".tsv", ".txt", ".parquet"):
        if stem.lower().endswith(ending):
            stem = stem[: -len(ending)]
            break
    if stem.lower() not in _GENERIC:
        return stem
    for parent in path.parents:
        label = parent.name
        if label.lower() not in _GENERIC and not re.fullmatch(r"square_\d+um", label.lower()) and label not in (".", ""):
            return label
    return "Section"


# ---------------------------------------------------------------- tables

def _sniff(path: Path) -> tuple[list[str], str]:
    with _open_text(path) as handle:
        first = handle.readline()
    if not first.strip():
        raise InputError(f"{path.name} is empty.")
    delimiter = "\t" if first.count("\t") > first.count(",") else ","
    columns = [c.strip() for c in next(csv.reader([first], delimiter=delimiter))]
    if any(not c for c in columns):
        raise InputError(f"{path.name} has an empty column name.")
    repeated = sorted({c for c in columns if columns.count(c) > 1})
    if repeated:
        raise InputError(f"{path.name} has repeated column names: {', '.join(repeated[:5])}.")
    return columns, delimiter


def _open_text(path: Path):
    raw = path.read_bytes()[:2]
    if raw == b"\x1f\x8b":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8-sig", newline="")
    return path.open(encoding="utf-8-sig", newline="")


def _read_plain_table(path: Path, name: str, text: Sequence[str] = TEXT_COLUMNS) -> pd.DataFrame:
    columns, delimiter = _sniff(path)
    try:
        with _open_text(path) as handle:
            frame = pd.read_csv(handle, sep=delimiter, header=0, names=columns, skipinitialspace=True,
                                dtype={c: str for c in text if c in columns}, keep_default_na=False,
                                na_values={c: MISSING for c in columns if c not in text})
    except Exception as error:
        raise InputError(f"{name} could not be read: {error}") from None
    if frame.empty:
        raise InputError(f"{name} has no data rows.")
    for column in text:
        if column in frame.columns:
            frame[column] = frame[column].str.strip()
    return frame


def _read_counts_table(path: Path, name: str, sample: str | None, um_per_unit: float | None) -> Counts:
    frame = _read_plain_table(path, name)
    if "unit_id" not in frame.columns:
        raise InputError(f"{name}: missing unit_id, the ID of each spot, bin or cell.")
    if "sample_id" not in frame.columns:
        frame.insert(0, "sample_id", sample or _section_name(path if name == path.name else Path(name)))
    for column in ("sample_id", "unit_id"):
        if frame[column].eq("").any():
            raise InputError(f"{name}: {column} cannot be empty.")
    if frame.duplicated(["sample_id", "unit_id"]).any():
        row = frame.loc[frame.duplicated(["sample_id", "unit_id"])].iloc[0]
        raise InputError(f"{name}: spot {row.sample_id} / {row.unit_id} appears more than once.")
    for column in ("x", "y"):
        if column not in frame.columns:
            raise InputError(f"{name}: x and y are needed, to place the spots, bins or cells.")
    genes = [c for c in frame.columns if c not in METADATA]
    if not genes:
        raise InputError(f"{name}: no gene columns.")
    values = _numbers(frame, genes, name)
    if np.isnan(values).any():
        raise InputError(f"{name}: counts cannot be missing. Write 0 where no molecule was counted.")
    _check_integers(values, name)
    xy = _numbers(frame, ["x", "y"], name)
    if not np.isfinite(xy).all():
        raise InputError(f"{name}: x or y is missing for some rows.")
    unit = "input"
    if um_per_unit:
        xy = xy * float(um_per_unit)
        unit = "um"
    extra = frame[[c for c in ("patient_id", "region", "cell_class", "area") if c in frame.columns]].copy()
    if "area" in extra.columns:
        extra["area"] = pd.to_numeric(extra["area"], errors="coerce") * (float(um_per_unit) ** 2 if um_per_unit else 1.0)
    return Counts(sparse.csr_matrix(values.astype(np.int64)), genes, frame["unit_id"].to_numpy(str),
                  frame["sample_id"].to_numpy(str), xy, unit, extra, None, name)


def _numbers(frame: pd.DataFrame, columns: list[str], name: str) -> np.ndarray:
    block = frame[columns]
    text = [c for c in columns if not pd.api.types.is_numeric_dtype(block[c])]
    if text:
        raise InputError(f"{name}: column {text[0]} has values that are not numbers.")
    return block.to_numpy(np.float64)


def _check_integers(values, name: str) -> None:
    data = values.data if sparse.issparse(values) else np.asarray(values)
    data = data[np.isfinite(data)] if data.dtype.kind == "f" else data
    if data.size and ((data < 0).any() or (data != np.rint(data)).any()):
        raise InputError(f"{name}: these are not raw counts. RSC splits molecules, so it needs whole-number counts from before any normalization or log.")


# ---------------------------------------------------------------- AnnData

def _h5py():
    try:
        import h5py
    except ImportError:  # pragma: no cover - depends on the installation
        raise InputError("Reading .h5ad and .h5 files needs h5py: pip install h5py") from None
    return h5py


def _decode(values) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in ("S", "O"):
        return np.asarray([v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values], dtype=object).astype(str)
    return values.astype(str)


def _h5_matrix(node) -> sparse.csr_matrix | np.ndarray:
    """An AnnData matrix: a dense dataset or a CSR/CSC group."""
    h5py = _h5py()
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[()])
    encoding = node.attrs.get("encoding-type", node.attrs.get("h5sparse_format", ""))
    encoding = encoding.decode() if isinstance(encoding, bytes) else str(encoding)
    shape = tuple(int(x) for x in node.attrs.get("shape", node.attrs.get("h5sparse_shape")))
    data, indices, indptr = node["data"][()], node["indices"][()], node["indptr"][()]
    if encoding.startswith("csc"):
        return sparse.csc_matrix((data, indices, indptr), shape=shape).tocsr()
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def _h5_frame_column(group, name: str) -> np.ndarray:
    """One column of an AnnData obs or var frame, decoding categoricals."""
    h5py = _h5py()
    node = group[name]
    if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
        categories = _decode(node["categories"][()])
        codes = node["codes"][()].astype(np.int64)
        out = np.where(codes >= 0, categories[np.clip(codes, 0, max(len(categories) - 1, 0))], "")
        return out.astype(object)
    values = node[()]
    if np.asarray(values).dtype.kind in ("S", "O"):
        return _decode(values).astype(object)
    return np.asarray(values)


def _h5_index(group) -> np.ndarray:
    key = group.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return _decode(group[key][()])


def _read_h5ad(path: Path, sample, layer, spatial_key, sample_column, columns, um_per_unit, original_name: str | None = None) -> Counts:
    h5py = _h5py()
    name = original_name or path.name
    with h5py.File(path, "r") as handle:
        if layer:
            if "layers" not in handle or layer not in handle["layers"]:
                raise InputError(f"{name}: no layer {layer!r}.")
            matrix, origin = _h5_matrix(handle["layers"][layer]), f"layers['{layer}']"
        elif "layers" in handle and "counts" in handle["layers"]:
            matrix, origin = _h5_matrix(handle["layers"]["counts"]), "layers['counts']"
        else:
            matrix, origin = _h5_matrix(handle["X"]), "X"
        obs = handle["obs"]
        ids = _h5_index(obs)
        var = handle["var"]
        genes = _h5_index(var)
        for symbol in ("gene_symbols", "gene_symbol", "feature_name"):
            if symbol in var:
                candidate = _h5_frame_column(var, symbol).astype(str)
                if len(set(candidate)) == len(candidate):
                    genes = candidate
                break
        xy, unit = None, "input"
        if "obsm" in handle and spatial_key in handle["obsm"]:
            coords = np.asarray(handle["obsm"][spatial_key][()], dtype=np.float64)
            xy = coords[:, :2]
        else:
            for xs, ys in (("x", "y"), ("x_centroid", "y_centroid"), ("x_um", "y_um"), ("array_col", "array_row")):
                if xs in obs and ys in obs:
                    xy = np.column_stack([_h5_frame_column(obs, xs).astype(float), _h5_frame_column(obs, ys).astype(float)])
                    if xs in ("x_centroid", "x_um"):
                        unit = "um"
                    break
        if xy is None:
            raise InputError(f"{name}: no coordinates. Give them in obsm['{spatial_key}'] or as obs columns x and y.")
        sections = np.full(len(ids), sample or _section_name(Path(name)), dtype=object)
        if sample_column:
            if sample_column not in obs:
                raise InputError(f"{name}: no obs column {sample_column!r}.")
            sections = _h5_frame_column(obs, sample_column).astype(str)
        elif "sample_id" in obs:
            sections = _h5_frame_column(obs, "sample_id").astype(str)
        wanted = list(columns or [])
        for default in ("region", "cell_class", "patient_id"):
            if default in obs and default not in wanted:
                wanted.append(default)
        extra = {}
        for column in wanted:
            if column not in obs:
                raise InputError(f"{name}: no obs column {column!r}.")
            extra[column] = _h5_frame_column(obs, column)
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
    else:
        matrix = np.asarray(matrix)
    _check_integers(matrix, f"{name} ({origin})")
    matrix = sparse.csr_matrix(matrix).astype(np.int64)
    keep = np.asarray([not any(str(g).startswith(p) for p in CONTROL_PREFIXES) for g in genes])
    if not keep.all():
        matrix, genes = matrix[:, np.flatnonzero(keep)].tocsr(), genes[keep]
    if um_per_unit:
        xy, unit = xy * float(um_per_unit), "um"
    return Counts(matrix, [str(g) for g in genes], ids.astype(str), np.asarray(sections, dtype=str), xy, unit,
                  pd.DataFrame(extra), None, f"{name} ({origin})")


# ---------------------------------------------------------------- 10x Genomics

_POSITION_NAMES = ("tissue_positions.parquet", "tissue_positions.csv", "tissue_positions_list.csv",
                   "cells.parquet", "cells.csv.gz", "cells.csv")
_MTX_MATRIX = ("matrix.mtx.gz", "matrix.mtx")
_MTX_FEATURES = ("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv")
_MTX_BARCODES = ("barcodes.tsv.gz", "barcodes.tsv")
_MTX_NAMES = _MTX_MATRIX + _MTX_FEATURES + _MTX_BARCODES


def _mtx_parts(folder: Path) -> dict[str, Path] | None:
    """The three files of a matrix-market folder of Space Ranger or Cell Ranger, if all are there."""
    found = {}
    for kind, names in (("matrix", _MTX_MATRIX), ("features", _MTX_FEATURES), ("barcodes", _MTX_BARCODES)):
        hit = next((folder / n for n in names if (folder / n).is_file()), None)
        if hit is None:
            return None
        found[kind] = hit
    return found


def _positions_next_to(matrix: Path) -> Path | None:
    for folder in (matrix.parent, matrix.parent / "spatial", matrix.parent.parent / "spatial"):
        for name in _POSITION_NAMES:
            if (folder / name).is_file():
                return folder / name
    return None


def _read_folder(path: Path, sample, layer, spatial_key, sample_column, columns, um_per_unit) -> Counts:
    for matrix_name in ("cell_feature_matrix.h5", "filtered_feature_bc_matrix.h5"):
        for candidate in (path / matrix_name, path / "outs" / matrix_name):
            if candidate.is_file():
                position = _positions_next_to(candidate)
                if position is None:
                    raise InputError(f"{candidate}: no tissue_positions or cells file next to it.")
                scale = next((p for p in (candidate.parent / "spatial" / "scalefactors_json.json",
                                          candidate.parent.parent / "spatial" / "scalefactors_json.json") if p.is_file()), None)
                return _read_10x(candidate, position, sample or _section_name(candidate), um_per_unit, scale)
    for candidate in (path, path / "filtered_feature_bc_matrix", path / "outs" / "filtered_feature_bc_matrix"):
        parts_ = _mtx_parts(candidate) if candidate.is_dir() else None
        if parts_:
            position = _positions_next_to(parts_["matrix"])
            if position is None:
                raise InputError(f"{candidate}: no tissue_positions file next to it or in the spatial folder beside it.")
            scale = next((p for p in (position.parent / "scalefactors_json.json", candidate.parent / "spatial" / "scalefactors_json.json")
                          if p.is_file()), None)
            return _read_10x_mtx(parts_, position, sample or _section_name(parts_["matrix"]), um_per_unit, scale)
    if (path / "binned_outputs").is_dir():
        options = sorted(p.name for p in (path / "binned_outputs").iterdir() if p.is_dir())
        raise InputError(f"{path.name}: Visium HD gives several bin sizes ({', '.join(options)}). Choose one of these folders.")
    found = sorted(p for p in path.iterdir() if p.is_dir())
    parts = []
    for sub in found:
        try:
            parts.append(_read_folder(sub, None, layer, spatial_key, sample_column, columns, um_per_unit))
        except InputError:
            continue
    files = sorted(p for p in path.iterdir() if p.is_file() and p.name.lower().endswith((".h5ad",) + TABLE_ENDINGS))
    for file in files:
        parts.append(read_counts(file, layer=layer, spatial_key=spatial_key, sample_column=sample_column, columns=columns, um_per_unit=um_per_unit))
    if not parts:
        raise InputError(f"{path.name}: no counts found. Expected a 10x matrix (.h5 or a filtered_feature_bc_matrix folder) with its positions, .h5ad files or CSV tables.")
    return concat(parts)


def _read_10x(matrix_path: Path, positions_path: Path, section: str, um_per_unit, scalefactors: Path | None = None,
              original_name: str | None = None) -> Counts:
    h5py = _h5py()
    name = original_name or matrix_path.name
    with h5py.File(matrix_path, "r") as handle:
        if "matrix" not in handle:
            raise InputError(f"{name}: not a 10x feature-barcode matrix.")
        group = handle["matrix"]
        shape = tuple(int(x) for x in group["shape"][()])
        csc = sparse.csc_matrix((group["data"][()], group["indices"][()], group["indptr"][()]), shape=shape)
        barcodes = _decode(group["barcodes"][()])
        features = group["features"]
        genes = _decode(features["name"][()])
        types = _decode(features["feature_type"][()]) if "feature_type" in features else np.full(len(genes), "Gene Expression")
    counts = csc.T.tocsr()  # barcodes x features
    return _finish_10x(counts, barcodes, genes, types, positions_path, section, um_per_unit, scalefactors, matrix_path, name)


def _read_10x_mtx(parts: dict[str, Path], positions_path: Path, section: str, um_per_unit, scalefactors: Path | None = None,
                  original_name: str | None = None) -> Counts:
    """A 10x matrix-market folder: ``matrix.mtx(.gz)``, ``features.tsv(.gz)`` (or ``genes.tsv`` of older
    versions) and ``barcodes.tsv(.gz)``."""
    from scipy import io as sio

    if sys.platform == "emscripten":  # pragma: no cover - WebAssembly cannot start threads; one thread reads the same numbers
        try:
            import scipy.io._fast_matrix_market as fast_matrix_market
            fast_matrix_market.PARALLELISM = 1
        except ImportError:
            pass
    name = original_name or parts["matrix"].parent.name + "/" + parts["matrix"].name
    opener = lambda p: gzip.open(p, "rb") if p.name.lower().endswith(".gz") else open(p, "rb")
    with opener(parts["matrix"]) as handle, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)   # SciPy's notice about sparse arrays
        matrix = sparse.csr_matrix(sio.mmread(handle))  # features x barcodes
    with opener(parts["features"]) as handle:
        features = pd.read_csv(handle, sep="\t", header=None, dtype=str)
    with opener(parts["barcodes"]) as handle:
        barcodes = pd.read_csv(handle, sep="\t", header=None, dtype=str)[0].to_numpy().astype(str)
    genes = features[1 if features.shape[1] > 1 else 0].to_numpy().astype(str)
    types = features[2].to_numpy().astype(str) if features.shape[1] > 2 else np.full(len(genes), "Gene Expression")
    if matrix.shape != (len(genes), len(barcodes)):
        raise InputError(f"{name}: the matrix is {matrix.shape[0]:,} × {matrix.shape[1]:,}, but there are {len(genes):,} features and {len(barcodes):,} barcodes.")
    counts = matrix.T.tocsr()
    if counts.dtype.kind == "f":
        if not np.all(np.mod(counts.data, 1) == 0):
            raise InputError(f"{name}: the matrix holds values that are not whole numbers.")
        counts = counts.astype(np.int64)
    return _finish_10x(counts, barcodes, genes, types, positions_path, section, um_per_unit, scalefactors, parts["matrix"], name)


def _finish_10x(counts, barcodes, genes, types, positions_path, section, um_per_unit, scalefactors, matrix_path, name) -> Counts:
    keep = (types == "Gene Expression") & np.asarray([not any(g.startswith(p) for p in CONTROL_PREFIXES) for g in genes])
    counts, genes = counts[:, np.flatnonzero(keep)].tocsr(), genes[keep]
    if len(set(genes)) != len(genes):
        counts, genes = _merge_duplicate_genes(counts, genes)
    _check_integers(counts, name)
    xy, ids, unit, spacing, extra = _positions(positions_path, barcodes, scalefactors, um_per_unit, matrix_path)
    rows = pd.Index(barcodes).get_indexer(ids)
    if (rows < 0).any():
        raise InputError(f"{positions_path.name}: {int((rows < 0).sum()):,} barcodes are not in {name}.")
    return Counts(counts[rows].tocsr().astype(np.int64), [str(g) for g in genes], ids.astype(str),
                  np.full(len(ids), section, dtype=object).astype(str), xy, unit, extra, spacing,
                  f"{name} with {positions_path.name}")


def _merge_duplicate_genes(counts: sparse.csr_matrix, genes: np.ndarray):
    unique, inverse = np.unique(genes, return_inverse=True)
    merge = sparse.csr_matrix((np.ones(len(genes)), (np.arange(len(genes)), inverse)), shape=(len(genes), len(unique)))
    return (counts @ merge).tocsr().astype(np.int64), unique


def _read_any_table(path: Path) -> pd.DataFrame:
    if path.name.lower().endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError:  # pragma: no cover
            raise InputError("Reading Parquet files needs pyarrow: pip install pyarrow") from None
        return pq.read_table(path).to_pandas()
    with _open_text(path) as handle:
        first = handle.readline()
    header = not first.split(",")[0].strip().endswith("-1") or "barcode" in first
    with _open_text(path) as handle:
        return pd.read_csv(handle, header=0 if header else None)


def _text(series: pd.Series) -> np.ndarray:
    """IDs as text; Parquet files can hold them as bytes."""
    return np.asarray([v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in series.to_numpy()], dtype=object).astype(str)


def _positions(path: Path, barcodes: np.ndarray, scalefactors: Path | None, um_per_unit, matrix_path: Path):
    frame = _read_any_table(path)
    lower = path.name.lower()
    extra = pd.DataFrame()
    if lower.startswith("cells"):
        need = {"cell_id", "x_centroid", "y_centroid"}
        if not need.issubset(frame.columns):
            raise InputError(f"{path.name}: expected the Xenium columns cell_id, x_centroid and y_centroid.")
        ids = _text(frame["cell_id"])
        present = pd.Index(barcodes)
        frame = frame.loc[present.get_indexer(ids) >= 0]
        ids = _text(frame["cell_id"])
        xy = frame[["x_centroid", "y_centroid"]].to_numpy(np.float64)
        return xy, ids, "um", None, extra
    if frame.columns[0] != "barcode":  # tissue_positions_list.csv of Space Ranger 1 has no header
        frame.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"][: frame.shape[1]]
    frame = frame.loc[frame["in_tissue"].astype(int) == 1]
    present = pd.Index(barcodes)
    frame = frame.loc[present.get_indexer(_text(frame["barcode"])) >= 0]
    ids = _text(frame["barcode"])
    if {"array_row", "array_col"}.issubset(frame.columns):   # the spot grid, for blocks of spots (strsc.regions)
        extra = pd.DataFrame({"array_row": frame["array_row"].to_numpy(np.int64), "array_col": frame["array_col"].to_numpy(np.int64)})
    factors = {}
    if scalefactors is None:
        for candidate in (path.parent / "scalefactors_json.json",):
            if candidate.is_file():
                scalefactors = candidate
    if scalefactors is not None:
        factors = json.loads(Path(scalefactors).read_text())
    bin_um = factors.get("bin_size_um")
    if bin_um is None:
        match = re.search(r"square_(\d+)um", str(matrix_path).lower())
        bin_um = float(match.group(1)) if match else None
    if bin_um:
        # Visium HD bins: the array indices are the grid, exactly
        xy = np.column_stack([frame["array_col"].to_numpy(np.float64), frame["array_row"].to_numpy(np.float64)]) * float(bin_um)
        return xy, ids, "um", float(bin_um), extra
    xy = np.column_stack([frame["pxl_col_in_fullres"].to_numpy(np.float64), frame["pxl_row_in_fullres"].to_numpy(np.float64)])
    if um_per_unit:
        return xy * float(um_per_unit), ids, "um", None, extra
    if factors.get("microns_per_pixel"):
        return xy * float(factors["microns_per_pixel"]), ids, "um", None, extra
    if factors.get("spot_diameter_fullres"):
        return xy * (VISIUM_SPOT_UM / float(factors["spot_diameter_fullres"])), ids, "um", None, extra
    return xy, ids, "input", None, extra


# ---------------------------------------------------------------- predictions

def read_predictions(path: str | Path, counts: Counts, name: str | None = None, *, original_name: str | None = None) -> Prediction:
    """Predicted expression from a CSV/TSV table or an .h5ad file, lined up with ``counts``.
    The model is named after the file unless ``name`` is given."""
    path = Path(path)
    shown = original_name or path.name
    model = name or _model_name(shown)
    lower = shown.lower()
    if lower.endswith(".h5ad"):
        h5py = _h5py()
        with h5py.File(path, "r") as handle:
            matrix = _h5_matrix(handle["X"])
            ids = _h5_index(handle["obs"])
            genes = _h5_index(handle["var"])
            sections = _h5_frame_column(handle["obs"], "sample_id").astype(str) if "sample_id" in handle["obs"] else None
        values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=np.float64)
        frame = pd.DataFrame(values, columns=[str(g) for g in genes])
        frame.insert(0, "unit_id", ids.astype(str))
        if sections is not None:
            frame.insert(0, "sample_id", sections)
    elif lower.endswith(TABLE_ENDINGS):
        frame = _read_plain_table(path, shown)
        if "unit_id" not in frame.columns:
            raise InputError(f"{shown}: missing unit_id.")
    else:
        raise InputError(f"{shown}: predictions are read from CSV or TSV tables, or .h5ad files.")
    genes = [c for c in frame.columns if c not in METADATA]
    present = [g for g in counts.genes if g in set(genes)]
    if not present:
        raise InputError(f"{shown}: none of its genes is in the counts. Use the same gene names.")
    index = _location_index(frame, counts, shown)
    values = np.full((len(counts), len(counts.genes)), np.nan)
    block = _numbers(frame, present, shown)
    columns = [counts.genes.index(g) for g in present]
    found = index >= 0
    values[np.ix_(index[found], columns)] = block[found]
    if not found.any():
        raise InputError(f"{shown}: none of its rows matches a spot, bin or cell of the counts. Match sample_id and unit_id.")
    return Prediction(model, values, present, shown)


def _model_name(name: str) -> str:
    base = Path(name).name
    for ending in (".csv.gz", ".tsv.gz", ".txt.gz", ".csv", ".tsv", ".txt", ".h5ad"):
        if base.lower().endswith(ending):
            base = base[: -len(ending)]
            break
    return base.strip() or "prediction"


def _location_index(frame: pd.DataFrame, counts: Counts, name: str) -> np.ndarray:
    """Row of the counts for each row of a table, matched on unit_id (and sample_id when given); -1 when absent."""
    sections = counts.section_names()
    if "sample_id" in frame.columns:
        keys = pd.MultiIndex.from_arrays([counts.sections.astype(str), counts.ids.astype(str)])
        wanted = pd.MultiIndex.from_arrays([frame["sample_id"].astype(str).to_numpy(), frame["unit_id"].astype(str).to_numpy()])
    else:
        if len(sections) > 1:
            raise InputError(f"{name}: the counts have several sections, so this file needs a sample_id column.")
        keys = pd.Index(counts.ids.astype(str))
        wanted = pd.Index(frame["unit_id"].astype(str).to_numpy())
    if wanted.has_duplicates:
        raise InputError(f"{name}: a spot, bin or cell appears more than once.")
    return np.asarray(keys.get_indexer(wanted))
