# PsiSpin

A spin-generalized fork of DeepMind's [FermiNet](https://github.com/deepmind/ferminet) (and MIT's [PeriodicWave](https://github.com/mg607/PeriodicWave)) neural-network VMC solver, aimed at condensed-matter systems.

Compared to upstream FermiNet, this fork:

- **Adds general treatment of spin degrees of freedom**, including MCMC moves that update spin (`cfg.mcmc.update_spin`), thus wavefunctions are not restricted to fixed spin sectors.
- **Adds spin-coupling physics**: such as the spin-momentum (`psispin/soc/`) and spin-position (aka Zeeman) coupling terms (`psispin/zeeman/`).
- **Removes quantum-chemistry features**: e.g. pseudo-potential and references to atomic positions. Streamlined for condensed matter application.

## Setup

```bash
python3 -m venv .venv --prompt="psispin"
source .venv/bin/activate
pip install "jax[cuda12]==0.7.2"
pip install -e .
pip install "kfac-jax==0.0.7"
```
We specified the versions of JAX and kfac for compatibility.

## Running

Local/interactive runs go through the `run_*.py` scripts directly (`psispin/soc/run_soc.py`, `psispin/zeeman/run_zeeman.py`, `psispin/pbc/run_pbc.py`), each taking positional arguments and writing results to `<folder_name>`.

For `run_soc.py`:

```bash
python3 psispin/soc/run_soc.py \
  <batch_size> <logging_pars> <network_type> <network_params> <optimizer> <learning_rate> \
  <use_spin_channel> <mcmc_steps> <update_spin> <spin_update_probability> <conserve_sz> \
  <kappa> <coulomb_strength> <electrons> <pbc_period> <folder_name>
```

| # | argument | type | meaning |
|---|---|---|---|
| 1 | `batch_size` | int | MCMC walker batch size |
| 2 | `logging_pars` | float | save/log frequency (in training steps) |
| 3 | `network_type` | str | `psiformer` or `SlaterNet` |
| 4 | `network_params` | str | psiformer only: `layers_heads_headsdim_mlphidden`, e.g. `4_4_32_128`; ignored for `SlaterNet` |
| 5 | `optimizer` | str | `adam`, `kfac`, `lamb`, or `none` |
| 6 | `learning_rate` | float | |
| 7 | `use_spin_channel` | bool-str | `True`/`False` |
| 8 | `mcmc_steps` | int | MCMC steps per training step |
| 9 | `update_spin` | bool-str | whether MCMC also proposes spin moves |
| 10 | `spin_update_probability` | float | e.g. `0.1` |
| 11 | `conserve_sz` | bool-str | `True`/`False` |
| 12 | `kappa` | str | `"none"`, or a JSON list like `[[0,1.0],[-1.0,0],[0,0]]` for the spin-orbit coupling matrix |
| 13 | `coulomb_strength` | float | Coulomb interaction strength |
| 14 | `electrons` | str | `up_down`, e.g. `2_0` for 2 spin-up, 0 spin-down |
| 15 | `pbc_period` | str | square lattice period, e.g. `6.0` (must be numeric, not `"none"`) |
| 16 | `folder_name` | str | output directory; created automatically if missing |

Example:

```bash
python3 psispin/soc/run_soc.py \
  2048 6.0 psiformer 4_4_32_128 kfac 0.01 \
  False 10 True 0.1 False \
  "[[0,1.0],[-1.0,0],[0,0]]" 0.0 5_0 6.0 results/soc_test/dos
```

`run_zeeman.py` and `run_pbc.py` follow the same pattern but with different parameter sets — see the top of each script for its exact `parse_args`/positional argument list.

## Giving credit

The corresponding PRB article is A Avdoshkin, M Geier, L Fu, "Integrated neural wave-function solver for spinful Fermi systems," Phys. Rev. B 113, 195108 (2026)
```
@article{avdoshkin2026,
  title = {Integrated neural wave-function solver for spinful Fermi systems},
  author = {Avdoshkin, Alexander and Geier, Max and Fu, Liang},
  journal = {Phys. Rev. B},
  volume = {113},
  issue = {19},
  pages = {195108},
  numpages = {8},
  year = {2026},
  month = {May},
  publisher = {American Physical Society},
  doi = {10.1103/9pwc-s37d},
  url = {https://link.aps.org/doi/10.1103/9pwc-s37d}
}

```
Please also see the list of related works in the repository https://github.com/mg607/PeriodicWave/ .
