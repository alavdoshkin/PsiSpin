
# Copyright 2020 DeepMind Technologies Limited.
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
# limitations under the License.

"""Super simple checkpoints using numpy."""

import dataclasses
import datetime
import os
from typing import Optional
import zipfile

from absl import logging
from psispin import networks
import jax
import jax.numpy as jnp
import numpy as np


def find_last_checkpoint(ckpt_path: Optional[str] = None,
         ckpt_identifier: str = "qmcjax_ckpt") -> Optional[str]:
  """Finds most recent valid checkpoint in a directory.

  Args:
    ckpt_path: Directory containing checkpoints.

  Returns:
    Last QMC checkpoint (ordered by sorting all checkpoints by name in reverse)
    or None if no valid checkpoint is found or ckpt_path is not given or doesn't
    exist. A checkpoint is regarded as not valid if it cannot be read
    successfully using np.load.
  """
  if ckpt_path and os.path.exists(ckpt_path):
    files = [f for f in os.listdir(ckpt_path) if ckpt_identifier + '_' in f]
    # Handle case where last checkpoint is corrupt/empty.
    for file in sorted(files, reverse=True):
      fname = os.path.join(ckpt_path, file)
      with open(fname, 'rb') as f:
        try:
          np.load(f, allow_pickle=True)
          return fname
        except (OSError, EOFError, zipfile.BadZipFile):
          logging.info('Error loading checkpoint %s. Trying next checkpoint...',
                       fname)
  return None


def create_save_path(save_path: Optional[str]) -> str:
  """Creates the directory for saving checkpoints, if it doesn't exist.

  Args:
    save_path: directory to use. If false, create a directory in the working
      directory based upon the current time.

  Returns:
    Path to save checkpoints to.
  """
  timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
  default_save_path = os.path.join(os.getcwd(), f'psispin_{timestamp}')
  ckpt_save_path = save_path or default_save_path
  if ckpt_save_path and not os.path.isdir(ckpt_save_path):
    os.makedirs(ckpt_save_path)
  return ckpt_save_path


def get_restore_path(restore_path: Optional[str] = None) -> Optional[str]:
  """Gets the path containing checkpoints from a previous calculation.

  Args:
    restore_path: path to checkpoints.

  Returns:
    The path or None if restore_path is falsy.
  """
  if restore_path:
    ckpt_restore_path = restore_path
  else:
    ckpt_restore_path = None
  return ckpt_restore_path


def save(save_path: str,
         t: int,
         data: networks.FermiNetData,
         params,
         opt_state,
         mcmc_width,
         ckpt_identifier: str = "qmcjax_ckpt") -> str:
  """Saves checkpoint information to a npz file.

  Args:
    save_path: path to directory to save checkpoint to. The checkpoint file is
      save_path/qmcjax_ckpt_$t.npz, where $t is the number of completed
      iterations.
    t: number of completed iterations.
    data: MCMC walker configurations.
    params: pytree of network parameters.
    opt_state: optimization state.
    mcmc_width: width to use in the MCMC proposal distribution.

  Returns:
    path to checkpoint file.
  """
  ckpt_filename = os.path.join(save_path, ckpt_identifier + f'_{t:06d}.npz')
  logging.info('Saving checkpoint %s', ckpt_filename)
  with open(ckpt_filename, 'wb') as f:
    np.savez(
        f,
        t=t,
        data=dataclasses.asdict(data),
        params=params,
        opt_state=np.asarray(opt_state, dtype=object),
        mcmc_width=mcmc_width)
  return ckpt_filename


def restore(restore_filename: str, batch_size: Optional[int] = None):
    """Restores data saved in a checkpoint.

    Args:
        restore_filename: filename containing checkpoint.
        batch_size: total batch size to be used. If present, check the data saved in
          the checkpoint is consistent with the batch size requested for the
          calculation.

    Returns:
        (t, data, params, opt_state, mcmc_width) tuple, where
        t: number of completed iterations.
        data: MCMC walker configurations.
        params: pytree of network parameters.
        opt_state: optimization state.
        mcmc_width: width to use in the MCMC proposal distribution.

    Raises:
        ValueError: if the leading dimension of data does not match the number of
        devices (i.e. the number of devices being parallelised over has changed) or
        if the total batch size is not equal to the number of MCMC configurations in
        data.
    """
    logging.info('Loading checkpoint %s', restore_filename)
    with open(restore_filename, 'rb') as f:
        ckpt_data = np.load(f, allow_pickle=True)
        # Retrieve data from npz file. Non-array variables need to be converted back
        # to native types using .tolist().
        t = ckpt_data['t'].tolist() + 1  # Return the iterations completed.
        data = networks.FermiNetData(**ckpt_data['data'].item())
        params = ckpt_data['params'].tolist()
        opt_state = ckpt_data['opt_state'].tolist()
        mcmc_width = jnp.array(ckpt_data['mcmc_width'].tolist())

        # Ensure number of devices matches
        if data.positions.shape[0] != jax.device_count():
            raise ValueError(
                f'Checkpoint data expects {data.positions.shape[0]} devices, '
                f'but found {jax.device_count()}.'
            )

        # Adjust batch size if necessary
        if batch_size is not None:
            original_batch_size = data.positions.shape[0] * data.positions.shape[1]
            if batch_size != original_batch_size:
                if batch_size % original_batch_size != 0:
                    raise ValueError(
                        f'New batch size {batch_size} is not an integer multiple of '
                        f'original batch size {original_batch_size}.'
                    )

                replication_factor = batch_size // original_batch_size
                logging.info(
                    f'Replicating data to match new batch size. Replication factor: {replication_factor}'
                )

                # Replicate data.positions along shape[1]
                data.positions = jnp.tile(data.positions, (1, replication_factor, 1))

                # Replicate other batch-dependent fields if necessary
                if data.spins is not None:
                    data.spins = jnp.tile(data.spins, (1, replication_factor, 1))

                logging.info(f'New positions shape: {data.positions.shape}')

    return t, data, params, opt_state, mcmc_width
