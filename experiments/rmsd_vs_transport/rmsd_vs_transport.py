#!/usr/bin/env python3
"""Plot RMSD and/or classical GW of PDB models relative to model 1."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

AtomKey = tuple[str, str, str, str]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent


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


def aligned_rmsd(reference: np.ndarray, coordinates: np.ndarray) -> float:
    """Return RMSD after optimal rotation and translation (Kabsch alignment)."""
    reference_centered = reference - reference.mean(axis=0)
    coordinates_centered = coordinates - coordinates.mean(axis=0)
    u, _, vh = np.linalg.svd(coordinates_centered.T @ reference_centered)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vh))
    rotation = u @ correction @ vh
    residual = reference_centered - coordinates_centered @ rotation
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def classical_gw(reference: np.ndarray, coordinates: np.ndarray) -> float:
    """Return the classical squared-loss GW objective at the identity coupling."""
    difference = (
        squared_distance_matrix(reference) - squared_distance_matrix(coordinates)
    )
    return float(np.mean(difference * difference))


def model_metrics(
    models: list[dict[AtomKey, np.ndarray]],
    *,
    compute_rmsd: bool,
    compute_classical_gw: bool,
    strict: bool = False,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Calculate requested metrics and matched-atom counts for each model."""
    reference = models[0]
    values: dict[str, list[float]] = {}
    if compute_rmsd:
        values["rmsd_angstrom"] = []
    if compute_classical_gw:
        values["classical_gw_angstrom4"] = []
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
        if compute_rmsd:
            values["rmsd_angstrom"].append(aligned_rmsd(ref_xyz, model_xyz))
        if compute_classical_gw:
            values["classical_gw_angstrom4"].append(
                classical_gw(ref_xyz, model_xyz)
            )
        counts.append(len(common))
    return {name: np.asarray(metric) for name, metric in values.items()}, counts


def parse_atom_names(values: Iterable[str]) -> set[str]:
    names = {name.strip().upper() for value in values for name in value.split(",")}
    names.discard("")
    if not names:
        raise ValueError("at least one atom name is required")
    return names


def resolve_input_path(value: str) -> Path:
    """Resolve input paths from the launch directory, then the script directory."""
    path = Path(to_absolute_path(value))
    if path.exists() or Path(value).is_absolute():
        return path
    return SCRIPT_DIRECTORY / value


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    if not config.rmsd and not config.classical_gw:
        raise ValueError("enable at least one metric: rmsd and/or classical_gw")

    atom_names = parse_atom_names(config.atoms)
    pdb_path = resolve_input_path(config.pdb)
    output_path = Path(to_absolute_path(config.output))
    csv_path = Path(to_absolute_path(config.csv)) if config.csv else None
    models = read_models(pdb_path, atom_names)
    if len(models) < 2:
        raise ValueError("the PDB must contain at least two models")
    metrics, counts = model_metrics(
        models,
        compute_rmsd=config.rmsd,
        compute_classical_gw=config.classical_gw,
        strict=config.strict,
    )
    indices = np.arange(1, len(models) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    axes = [ax]
    if config.rmsd:
        ax.plot(indices, metrics["rmsd_angstrom"], marker="o", label="RMSD")
        ax.set_ylabel(r"RMSD ($\mathrm{\AA}$)")
    if config.classical_gw:
        gw_ax = ax.twinx() if config.rmsd else ax
        if gw_ax is not ax:
            axes.append(gw_ax)
        gw_ax.plot(
            indices,
            metrics["classical_gw_angstrom4"],
            marker="s",
            color="tab:orange",
            label="Classical GW",
        )
        gw_ax.set_ylabel(r"Identity-coupling classical GW ($\mathrm{\AA}^4$)")
    ax.set_xlabel("Model index")
    ax.set_xticks(indices if len(indices) <= 20 else ax.get_xticks())
    ax.grid(alpha=0.3)
    if len(metrics) > 1:
        lines = [line for axis in axes for line in axis.lines]
        ax.legend(lines, [line.get_label() for line in lines])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    if csv_path:
        metric_names = list(metrics)
        data = np.column_stack(
            [indices, *(metrics[name] for name in metric_names), counts]
        )
        np.savetxt(
            csv_path,
            data,
            delimiter=",",
            header=",".join(["model", *metric_names, "n_atoms"]),
            comments="",
            fmt=["%d", *(["%.8f"] * len(metric_names)), "%d"],
        )
    print(f"Compared {len(models)} models using {','.join(sorted(atom_names))}")
    print(f"Matching atoms per model: {min(counts)}-{max(counts)}")
    print(f"Saved plot to {output_path}")
    if csv_path:
        print(f"Saved values to {csv_path}")
    if config.show:
        plt.show()


if __name__ == "__main__":
    main()
