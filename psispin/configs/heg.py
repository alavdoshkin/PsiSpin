# Copyright 2022 DeepMind Technologies Limited.
# Modifications Copyright (c) 2026 Alexander Avdoshkin, Max Geier, Massachusetts Institute of Technology, MA, USA
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
# limitations under the License

"""Unpolarised 14 electron simple cubic homogeneous electron gas."""

from psispin import base_config
from psispin.pbc import envelopes

import numpy as np


def _sc_lattice_vecs(rs: float, nelec: int) -> np.ndarray:
  """Returns simple cubic lattice vectors with Wigner-Seitz radius rs."""
  volume = (4 / 3) * np.pi * (rs**3) * nelec
  length = volume**(1 / 3)
  return length * np.eye(3)


def get_config():
  """Returns config for running unpolarised 14 electron gas with FermiNet."""
  # Get default options.
  cfg = base_config.default()
  cfg.system.electrons = (7, 7)
  lattice = _sc_lattice_vecs(1.0, sum(cfg.system.electrons))
  kpoints = envelopes.make_kpoints(lattice, cfg.system.electrons)

  cfg.system.make_local_energy_fn = "psispin.pbc.hamiltonian.local_energy"
  cfg.system.make_local_energy_kwargs = {"lattice": lattice, "heg": True}
  cfg.network.make_feature_layer_fn = (
      "psispin.pbc.feature_layer.make_pbc_feature_layer")
  cfg.network.make_feature_layer_kwargs = {
      "lattice": lattice,
  }
  cfg.network.make_envelope_fn = (
      "psispin.pbc.envelopes.make_multiwave_envelope")
  cfg.network.make_envelope_kwargs = {"kpoints": kpoints}
  cfg.network.full_det = True
  return cfg
