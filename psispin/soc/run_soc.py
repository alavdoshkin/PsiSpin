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

import argparse

from absl import logging
from psispin.pbc import envelopes
from psispin import base_config
from psispin import train

from psispin.utils import writers

from typing import Tuple

from time import time

import jax.numpy as jnp
import jax

def str2bool(v: str) -> bool:
    # Accept common truthy/falsey strings: true/false/1/0/yes/no
    if isinstance(v, bool):
        return v
    s = v.strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("batch_size", type=int)
    p.add_argument("logging_pars", type=float)
    p.add_argument("network_type", type=str)
    p.add_argument("network_params", type=str, help='e.g. "4_8_16" or "${num_layers}_${num_heads}_${mlp_hidden_dims}"')
    p.add_argument("optimizer", type=str)
    p.add_argument("learning_rate", type=float)
    p.add_argument("use_spin_channel", type=str2bool)
    p.add_argument("mcmc_steps", type=int)
    p.add_argument("update_spin", type=str2bool)
    p.add_argument("spin_update_probability", type=float)  # (spelled correctly)
    p.add_argument("conserve_sz", type=str2bool)
    p.add_argument("kappa", type=str)
    p.add_argument("coloumb_strength", type=float)
    p.add_argument("electrons", type=str, help='e.g. "2_0" for 2 spin up, 0 spin down')
    p.add_argument("pbc_period", type=str, help='e.g. "none" or "3.5"')
    p.add_argument("folder_name", type=str)
    return p.parse_args(argv)


def _sq_lattice_vecs(period: float) -> np.ndarray:
  """Returns simple square lattice vectors"""
  return period * np.eye(2)

# Get parameters defining the physical system
print(argv)

#python3 psispin/harmonic/run_SOC.py $batch_size $logging_pars $learning_rate $spin_update_prob $folder_name" > $job_file

# program_name = argv[0]
# batch_size = float(argv[1])
# logging_pars = float(argv[2])
# network_type = argv[3]  # e.g. "SlaterNet", "psiformer"
# network_params = argv[4]  # e.g. "4_8_16", "${num_layers}_${num_heads}_${mlp_hidden_dims}"
# learning_rate = float(argv[5])
# use_spin_channel = (argv[6]=='True')   
# update_spin = (argv[7] == 'True')  
# spin_update_probality = float(argv[8]) 
# conserve_sz = (argv[9]=='True')
# kappa = argv[10]
# coloumb_strength = float(argv[11])

# electrons = argv[12]  # e.g. "2_0" for 2 spin up, 0 spin down
# pbc_period = argv[13]  # e.g. "none" or "3.5" for PBC with period 3.5


# # Get folder name to save the results
# folder_name = argv[14]

args = parse_args()


batch_size = args.batch_size
logging_pars = args.logging_pars
network_type = args.network_type
network_params = args.network_params
optimizer = args.optimizer
learning_rate = args.learning_rate
use_spin_channel = args.use_spin_channel
mcmc_steps = args.mcmc_steps
update_spin = args.update_spin
spin_update_probability = args.spin_update_probability
conserve_sz = args.conserve_sz
kappa = args.kappa
coloumb_strength = args.coloumb_strength
electrons = args.electrons
pbc_period = args.pbc_period
folder_name = args.folder_name


# --------------------------- Set up config file ---------------------------
cfg = base_config.default()

cfg.system.ndim = 2

cfg.network.make_envelope_fn = ( "psispin.envelopes.make_null_envelope" )


up_electrons, down_electrons = [int(x) for x in electrons.split('_')]
cfg.system.electrons = (up_electrons, down_electrons)

cfg.system.make_local_energy_fn = "psispin.soc.hamiltonian.local_energy"

cfg.optim.optimizer = optimizer

cfg.optim.lr.rate = learning_rate



if network_type == 'psiformer':
    cfg.network.network_type = 'psiformer'
    cfg.network.complex = True
    cfg.network.determinants = 4
    cfg.network.psiformer.num_layers = int(network_params.split('_')[0])
    cfg.network.psiformer.num_heads = int(network_params.split('_')[1])
    cfg.network.psiformer.heads_dim = int(network_params.split('_')[2])
    cfg.network.psiformer.mlp_hidden_dims = (int(network_params.split('_')[3]),)
elif network_type == 'SlaterNet':
    cfg.network.complex = True
    cfg.network.network_type = 'SlaterNet'
    cfg.network.SlaterNet.num_layers = 4
    cfg.network.SlaterNet.mlp_dim = 128
    cfg.network.determinants = 1
    cfg.network.SlaterNet.num_perceptrons_per_layer = 2
    cfg.network.SlaterNet.use_layer_norm = True
    cfg.network.SlaterNet.mlp_activation_fct = "GELU"
else:
    raise ValueError(f"Unknown network type: {network_type}")
cfg.use_spin_channels = use_spin_channel

cfg.mcmc.update_spin = update_spin
cfg.mcmc.spin_update_probability = spin_update_probability
cfg.mcmc.conserve_sz = conserve_sz

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

cfg.mcmc.steps = mcmc_steps

#kappa = [[0,1],[-1,0],[0,0]]
if kappa != 'none':
    kappa = jnp.array(json.loads(kappa))
    cfg.system.make_local_energy_kwargs = {"lattice": lattice, "potential_kwargs": {'laplacian_method': "default",'kappa_soc' : kappa} }
else:
    cfg.system.make_local_energy_kwargs = {"lattice": lattice, "potential_kwargs": {'laplacian_method': "default"}}

cfg.system.make_local_energy_kwargs['potential_kwargs']['coloumb_strength'] = coloumb_strength

os.makedirs(folder_name, exist_ok=True)

outputfile = f"{folder_name}/custum_output"
with open(outputfile, 'w') as f:
     f.write(f"Batch size: {batch_size}\n")
     f.write(f"Network configuration: {network_type}\n")
     f.write(f"Electrons: {electrons}\n")
     f.write(f"network type: {network_type}\n")
     f.write(f"network params: {network_params}\n")
     f.write(f"pbc_period: {pbc_period}\n")
     f.write(f"lattice: {lattice}\n")
     f.write(f"Folder name: {folder_name}\n")
     f.write(f"Use spin channel: {use_spin_channel}\n") 
     f.write(f"Update spin: {update_spin}\n")
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