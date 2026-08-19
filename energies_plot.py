# Copyright (c) 2025 Alexander Avdoshkin, Massachusetts Institute of Technology, MA, USA
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

"""Plots a training/inference energy trace using psispin.utils.analysis_tools.

Unlike energies_plot.py (which regex-parses raw stdout logs), this reads the
structured train_stats.csv / inference_stats.csv written by psispin.train,
and uses pyblock reblocking (via analysis_tools.estimate_stats) to report a
proper correlation-corrected mean and standard error for the post-burn-in
energy, rather than eyeballing a moving average.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psispin.utils import analysis_tools


def moving_average(x, window):
    x = np.asarray(x, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    if window > len(x):
        raise ValueError("window cannot be larger than the data length")
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_energy(csv_path, burn_in=None, window=200, stride=1, exact=None):
    """Plots the energy trace and overlays a reblocked equilibrium estimate."""
    df = pd.read_csv(csv_path)

    if burn_in is None:
        burn_in = len(df) // 2

    stats = analysis_tools.estimate_stats(
        df.rename(columns={'energy': 'eigenvalues'}).assign(group=0),
        burn_in=burn_in,
        groups=['group'],
        group_by_work_unit=False,
    )
    mean = stats['energy'].iloc[0]
    stderr = stats['stderr'].iloc[0]

    steps = df['step'].to_numpy()
    energy = df['energy'].to_numpy()
    energy_ma = moving_average(energy, window)
    steps_ma = steps[window - 1:]

    plt.figure()
    plt.plot(steps[::stride], energy[::stride], alpha=0.25, lw=0.5,
              label='energy (raw)')
    plt.plot(steps_ma[::stride], energy_ma[::stride],
              label=f'moving average (w={window})')
    plt.axvline(steps[burn_in], color='gray', ls=':', label=f'burn-in={burn_in}')
    plt.axhline(mean, color='k', ls='--',
                label=f'reblocked estimate: {mean:.5f} ± {stderr:.5f}')
    plt.axhspan(mean - stderr, mean + stderr, color='k', alpha=0.15)
    if exact is not None:
        plt.axhline(exact, color='red', ls=':', lw=2,
                    label=f'exact (non-interacting): {exact:.5f}')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.legend()
    plt.tight_layout()

    out_path = csv_path.rsplit('.', 1)[0] + '_energy_plot.png'
    plt.savefig(out_path)
    msg = f'{csv_path}: mean = {mean:.6f}, stderr = {stderr:.6f} (burn_in={burn_in}/{len(df)})'
    if exact is not None:
        msg += f', exact = {exact:.6f}, diff = {mean - exact:.6f}'
    print(msg)
    print(f'saved plot to {out_path}')
    return out_path


if __name__ == '__main__':
    path = sys.argv[1]
    burn_in_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    window_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    exact_arg = float(sys.argv[4]) if len(sys.argv) > 4 else None
    plot_energy(path, burn_in=burn_in_arg, window=window_arg, exact=exact_arg)
