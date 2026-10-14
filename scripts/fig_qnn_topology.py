"""Draw a QNN topology from a solution file as a circuit diagram.

    python scripts/fig_qnn_topology.py \
        path/to/qnn_circuit_topology.py \
        qnn_topology.png \
        --title "sheldon: score 1.0"

The solution file is the artifact the run submitted: a `build_topology()`
returning a list of gates. Gates are packed into the earliest column their
qubits are free in, so the picture shows depth as well as order.
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

INK = "#1f2933"
ROTATION = {"RX": "#dbeafe", "RY": "#dcfce7", "RZ": "#fef3c7"}
OTHER = "#f1f5f9"


def columns(topology: list[dict]) -> list[tuple[int, dict]]:
    """Each gate with the earliest column in which all its qubits are free."""
    free: dict[int, int] = {}
    placed = []
    for gate in topology:
        qubits = [gate[k] for k in ("qubit", "control", "target") if k in gate]
        col = max((free.get(q, 0) for q in qubits), default=0)
        lo, hi = min(qubits), max(qubits)
        for q in range(lo, hi + 1):
            free[q] = col + 1
        placed.append((col, gate))
    return placed


def draw(topology: list[dict], out: Path, title: str | None) -> None:
    placed = columns(topology)
    n_qubits = 1 + max(g[k] for _, g in placed for k in ("qubit", "control", "target") if k in g)
    n_cols = 1 + max(col for col, _ in placed)

    fig, ax = plt.subplots(figsize=(1.1 * n_cols + 2.2, 1.1 * n_qubits + 1.3), dpi=200)
    y = lambda q: n_qubits - 1 - q  # noqa: E731  qubit 0 on top

    for q in range(n_qubits):
        ax.plot([-0.8, n_cols - 0.2], [y(q), y(q)], color=INK, lw=1.2, zorder=1)
        ax.text(-1.0, y(q), f"$q_{q}$", ha="right", va="center", fontsize=14, color=INK)

    for col, gate in placed:
        name = gate["gate"]
        if name == "CNOT":
            c, t = y(gate["control"]), y(gate["target"])
            ax.plot([col, col], [c, t], color=INK, lw=1.2, zorder=2)
            ax.add_patch(Circle((col, c), 0.08, color=INK, zorder=3))
            ax.add_patch(Circle((col, t), 0.2, facecolor="white", edgecolor=INK, lw=1.2, zorder=3))
            ax.plot([col - 0.2, col + 0.2], [t, t], color=INK, lw=1.2, zorder=4)
            ax.plot([col, col], [t - 0.2, t + 0.2], color=INK, lw=1.2, zorder=4)
        else:
            q = y(gate["qubit"])
            ax.add_patch(
                FancyBboxPatch(
                    (col - 0.3, q - 0.3), 0.6, 0.6,
                    boxstyle="round,pad=0,rounding_size=0.06",
                    facecolor=ROTATION.get(name, OTHER), edgecolor=INK, lw=1.2, zorder=3,
                )
            )
            label = f"$R_{name[1].lower()}$" if name in ROTATION else name
            ax.text(col, q, label, ha="center", va="center", fontsize=13, color=INK, zorder=4)

    n_param = sum(g["gate"] in ROTATION for _, g in placed)
    ax.text(
        -0.8, -0.75,
        f"{len(placed)} gates, depth {n_cols}, {n_param} trainable rotations",
        ha="left", va="center", fontsize=10, color="#52606d",
    )
    if title:
        ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
    ax.set_xlim(-1.5, n_cols - 0.1)
    ax.set_ylim(-1.05, n_qubits - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("solution", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--title")
    args = ap.parse_args()
    topology = runpy.run_path(str(args.solution))["build_topology"]()
    draw(topology, args.out, args.title)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
