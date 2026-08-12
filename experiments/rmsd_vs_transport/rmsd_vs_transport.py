#!/usr/bin/env python3
"""Plot identity-plan GW discrepancy of PDB models relative to model 1."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AtomKey = tuple[str, str, str, str]


def read_models(path: Path, atom_names: set[str]) -> list[dict[AtomKey, np.ndarray]]:
    """Read selected atoms from MODEL records (or one implicit model)."""
    models: list[dict[AtomKey, np.ndarray]] = []
    current: dict[AtomKey, np.ndarray] = {}
    saw_model = False

    def finish_model() -> None:
        nonlocal current
        if current:
            models.append(current)
        current = {}

    with path.open() as pdb:
        for line in pdb:
            record = line[:6].strip()
            if record == "MODEL":
                if saw_model:
                    finish_model()
                saw_model = True
                continue
            if record == "ENDMDL":
                finish_model()
                continue
            if record not in {"ATOM", "HETATM"}:
                continue

            name = line[12:16].strip()
            altloc = line[16:17]
            if name not in atom_names or altloc not in {" ", "A"}:
                continue

            key = (line[21:22].strip(), line[22:26].strip(), line[26:27].strip(), name)
            # Prefer the blank (primary) conformer if both blank and A occur.
            if key in current and altloc == "A":
                continue
            try:
                current[key] = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
            except ValueError as exc:
                raise ValueError(f"Invalid coordinates in {path}: {line.rstrip()}") from exc

    finish_model()
    if not models:
        raise ValueError(f"No selected atoms found in {path}")
    return models


def squared_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """Return the matrix whose (i, j) entry is ||x_i - x_j||^2."""
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    return np.einsum("ijk,ijk->ij", differences, differences)


def model_discrepancies(
    models: list[dict[AtomKey, np.ndarray]], strict: bool = False
) -> tuple[np.ndarray, list[int]]:
    """Calculate uniform identity-plan GW objectives and atom counts.

    For n matched atoms this is
    (1 / n^2) * sum_ij (||x_i-x_j||^2 - ||y_i-y_j||^2)^2.
    """
    reference = models[0]
    discrepancies: list[float] = []
    counts: list[int] = []
    for index, model in enumerate(models, start=1):
        common = sorted(reference.keys() & model.keys())
        if strict and model.keys() != reference.keys():
            missing = len(reference.keys() - model.keys())
            extra = len(model.keys() - reference.keys())
            raise ValueError(
                f"Model {index} differs from model 1 (missing={missing}, extra={extra})"
            )
        if len(common) < 3:
            raise ValueError(f"Model {index} has only {len(common)} matching selected atoms")
        ref_xyz = np.stack([reference[key] for key in common])
        model_xyz = np.stack([model[key] for key in common])
        difference = (
            squared_distance_matrix(ref_xyz) - squared_distance_matrix(model_xyz)
        )
        discrepancies.append(float(np.mean(difference * difference)))
        counts.append(len(common))
    return np.asarray(discrepancies), counts


def parse_atom_names(values: Iterable[str]) -> set[str]:
    names = {name.strip().upper() for value in values for name in value.split(",")}
    names.discard("")
    if not names:
        raise argparse.ArgumentTypeError("at least one atom name is required")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb", type=Path, help="multi-model PDB file")
    parser.add_argument(
        "--atoms", nargs="+", default=["N", "CA", "C", "O"],
        help="atom names, space- or comma-separated (default: N CA C O)",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("identity_gw.png"))
    parser.add_argument(
        "--csv", type=Path, help="optionally save model,discrepancy,n_atoms"
    )
    parser.add_argument("--strict", action="store_true", help="require identical atom sets")
    parser.add_argument("--show", action="store_true", help="show the interactive plot")
    args = parser.parse_args()

    atom_names = parse_atom_names(args.atoms)
    models = read_models(args.pdb, atom_names)
    if len(models) < 2:
        parser.error("the PDB must contain at least two models")
    discrepancies, counts = model_discrepancies(models, strict=args.strict)
    indices = np.arange(1, len(models) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(indices, discrepancies, marker="o", linewidth=1.5)
    ax.set(
        xlabel="Model index",
        ylabel=r"Identity-plan GW discrepancy ($\mathrm{\AA}^4$)",
    )
    ax.set_xticks(indices if len(indices) <= 20 else ax.get_xticks())
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)

    if args.csv:
        data = np.column_stack((indices, discrepancies, counts))
        np.savetxt(
            args.csv,
            data,
            delimiter=",",
            header="model,identity_gw_angstrom4,n_atoms",
            comments="",
            fmt=["%d", "%.8f", "%d"],
        )
    print(f"Compared {len(models)} models using {','.join(sorted(atom_names))}")
    print(f"Matching atoms per model: {min(counts)}-{max(counts)}")
    print(f"Saved plot to {args.output}")
    if args.csv:
        print(f"Saved values to {args.csv}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
