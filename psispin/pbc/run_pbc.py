# Copyright (c) 2026 Alexander Avdoshkin, Massachusetts Institute of Technology, MA, USA
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

import os
import sys
from sys import argv
import numpy as np
import json


from absl import logging
from psispin.pbc import envelopes
from psispin import base_config
from psispin import train

from psispin.utils import writers

from typing import Tuple

from time import time

import jax.numpy as jnp
import jax

def _sq_lattice_vecs(period: float) -> np.ndarray:
  """Returns simple square lattice vectors"""
  return period * np.eye(2)

# Get parameters defining the physical system
print(argv)

#python3 psispin/harmonic/run_SOC.py $batch_size $logging_pars $learning_rate $spin_update_prob $folder_name" > $job_file

program_name = argv[0]
batch_size = float(argv[1])
logging_pars = float(argv[2])
network = argv[3]  # e.g. "4_8_16", "${num_layers}_${num_heads}_${mlp_hidden_dims}"
learning_rate = float(argv[4])
electrons = argv[5]  # e.g. "2_0" for 2 spin up, 0 spin down
pbc_period = argv[6]  # e.g. "none" or "3.5" for PBC with period 3.5


# Get folder name to save the results
folder_name = argv[7]

# --------------------------- Set up config file ---------------------------
cfg = base_config.default()

cfg.system.ndim = 2

cfg.network.make_envelope_fn = ( "psispin.envelopes.make_null_envelope" )


up_electrons, down_electrons = [int(x) for x in electrons.split('_')]
cfg.system.electrons = (up_electrons, down_electrons)

cfg.system.make_local_energy_fn = "psispin.pbc.hamiltonian.local_energy"


cfg.network.network_type = 'psiformer'

cfg.optim.lr.rate = learning_rate

cfg.network.complex = True
cfg.network.determinants = 4

cfg.network.psiformer.num_layers = int(network.split('_')[0])
cfg.network.psiformer.num_heads = int(network.split('_')[1])
cfg.network.psiformer.heads_dim = int(network.split('_')[2])
cfg.network.psiformer.mlp_hidden_dims = (int(network.split('_')[3]),)


if pbc_period == 'none':
    pass
else:
    pbc_period = float(pbc_period)
    lattice = _sq_lattice_vecs(pbc_period)
    cfg.network.make_feature_layer_fn = (
        "psispin.pbc.feature_layer.make_pbc_feature_layer")
    cfg.network.make_feature_layer_kwargs = {
        "lattice": lattice,
    }

cfg.system.pbc_lattice = lattice
cfg.system.make_local_energy_kwargs = {"lattice": lattice, "potential_kwargs": {'laplacian_method': "default"}}

outputfile = f"{folder_name}/custum_output"
with open(outputfile, 'w') as f:
     f.write(f"Batch size: {batch_size}\n")
     f.write(f"Network configuration: {network}\n")
     f.write(f"Electrons: {electrons}\n")
     f.write(f"pbc_period: {pbc_period}\n")
     f.write(f"lattice: {lattice}\n")
     f.write(f"Folder name: {folder_name}\n")
     f.close()


cfg.network.full_det = True
cfg.log.save_frequency = logging_pars

# jax.config.update("jax_default_matmul_precision", "float32")
# print(f"Matmul precision set to: {jax.default_matmul_precision}")

# save path
cfg.log.save_path = folder_name

t_init = time()
# --------------------------- train ---------------------------
train.train(cfg)

logging.info("Training completed after t [s] = " + str(int(time() - t_init)))