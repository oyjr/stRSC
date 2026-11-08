"""Frozen, endpoint-blind prediction manifest for DeepSpot and HiST."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import (
    MODELS,
    RHOS,
    SEEDS,
    SOURCE_REVISIONS,
    TASK_SPECIMENS,
    canonical_model_name,
)


MANIFEST_SCHEMA = "g0055_hest_deepspot_hist_prediction_manifest_v1"


def build_prediction_manifest(
    *,
    prediction_root: str | Path,
    models: Sequence[str] = MODELS,
    source_revisions: Mapping[str, str] = SOURCE_REVISIONS,
) -> dict[str, Any]:
    root = Path(prediction_root)
    rows: list[dict[str, Any]] = []
    for requested in models:
        model = canonical_model_name(requested)
        revision = str(source_revisions[model])
        for task, specimens in TASK_SPECIMENS.items():
            for fold, specimen in enumerate(specimens):
                for seed in SEEDS:
                    for rho in RHOS:
                        rho_tag = str(rho).replace(".", "p")
                        output = root / model / task / (
                            f"fold{fold}.{specimen}.seed{seed}.rho{rho_tag}.h5"
                        )
                        rows.append(
                            {
                                "model": model,
                                "task": task,
                                "fold": fold,
                                "specimen": specimen,
                                "seed": seed,
                                "rho": rho,
                                "source_revision": revision,
                                "prediction_path": str(output),
                                "status": "PLANNED",
                            }
                        )
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "FROZEN_PLANNED",
        "scientific_results_opened": False,
        "models": [canonical_model_name(value) for value in models],
        "rows": rows,
    }


def validate_prediction_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("PREDICTION_MANIFEST_SCHEMA")
    rows = list(payload.get("rows", ()))
    models = tuple(canonical_model_name(value) for value in payload.get("models", ()))
    expected = len(models) * sum(len(values) for values in TASK_SPECIMENS.values()) * len(SEEDS) * len(RHOS)
    if len(rows) != expected:
        raise ValueError(f"PREDICTION_MANIFEST_ROWS:{len(rows)}:{expected}")
    keys = [
        (row["model"], row["task"], int(row["fold"]), row["specimen"], int(row["seed"]), float(row["rho"]))
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("PREDICTION_MANIFEST_DUPLICATE")
    if payload.get("scientific_results_opened") is not False:
        raise ValueError("PREDICTION_MANIFEST_ENDPOINT_FLAG")


def write_prediction_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    validate_prediction_manifest(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=MODELS)
    args = parser.parse_args()
    payload = build_prediction_manifest(
        prediction_root=args.prediction_root,
        models=tuple(args.model) if args.model else MODELS,
    )
    write_prediction_manifest(args.output, payload)
    print(json.dumps({"status": payload["status"], "rows": len(payload["rows"])}))


if __name__ == "__main__":
    main()
