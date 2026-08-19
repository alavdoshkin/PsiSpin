# Copyright (c) 2025 Alexander Avdoshkin, Max Geier, Massachusetts Institute of Technology, MA, USA
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Plots electron density from qmcjax_ckpt_*.npz checkpoints for SOC/SS runs.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _sq_lattice_vecs(period: float) -> np.ndarray:
    """Returns simple square lattice vectors (matches run_soc.py/run_zeeman.py)."""
    return period * np.eye(2)


def reciprocal_lattice(lattice: np.ndarray) -> np.ndarray:
    return 2 * np.pi * np.linalg.inv(lattice).T


def send_positions_to_first_unit_cell(
    positions: np.ndarray, lattice: np.ndarray, rec: np.ndarray
) -> np.ndarray:
    """Folds positions into the PBC cell centered at the origin."""
    phase = np.einsum('il,kl->ki', rec / (2 * np.pi), positions)
    phase_prim = (phase + 0.5) % 1 - 0.5
    return np.einsum('il,kl->ki', lattice, phase_prim)


def load_positions_and_spins(folder: str, ndim: int, num_checkpoints: int):
    """Loads walker positions/spins from the last `num_checkpoints` npz files."""
    cpts = sorted(f for f in os.listdir(folder) if f.lower().endswith('.npz'))
    cpts = cpts[-num_checkpoints:]

    collected_pos = np.empty((0, ndim))
    collected_spins = np.empty((0,))
    for cpt in cpts:
        data = np.load(os.path.join(folder, cpt), allow_pickle=True)
        state = data['data'].item()
        pos = state['positions'][0].copy().reshape(-1, ndim)
        spins = state['spins'][0].copy().reshape(-1)
        collected_pos = np.concatenate((collected_pos, pos))
        collected_spins = np.concatenate((collected_spins, spins))
    return collected_pos, collected_spins, cpts


def plot_density(folder: str, pbc_period: float, ndim: int = 2,
                  num_checkpoints: int = 30, n_cells: int = 2,
                  outpath: str = None):
    """Scatter-plots folded spin-up/spin-down density inside the SOC/SS PBC cell."""
    lattice = _sq_lattice_vecs(pbc_period)
    rec = reciprocal_lattice(lattice)

    pos, spins, cpts = load_positions_and_spins(folder, ndim, num_checkpoints)
    pos = send_positions_to_first_unit_cell(pos, lattice, rec)

    pos_up = pos[spins == 1]
    pos_down = pos[spins != 1]
    print(f"Total up: {len(pos_up)}, total down: {len(pos_down)}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pos_up[:, 0], pos_up[:, 1], c='tab:red', label='Up', s=0.3, alpha=0.6)
    ax.scatter(pos_down[:, 0], pos_down[:, 1], c='tab:blue', label='Down', s=0.3, alpha=0.6)

    half = pbc_period / 2
    ax.add_patch(patches.Rectangle((-half, -half), pbc_period, pbc_period,
                                    fill=False, ls='--', lw=1.0, edgecolor='black'))

    lim = n_cells * half
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(markerscale=10)
    ax.set_title(f'Density, {cpts[-1]}' if cpts else 'Density')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    outpath = outpath or os.path.join(folder, 'density.png')
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved density plot to: {outpath}")
    return outpath


if __name__ == "__main__":
    # Edit these to point at a results folder produced by run_soc.py / run_zeeman.py.
    folder = "results/my_run"
    pbc_period = 6.0
    num_checkpoints = 30

    plot_density(folder, pbc_period, num_checkpoints=num_checkpoints)
