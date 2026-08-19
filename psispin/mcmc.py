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

"""Metropolis-Hastings Monte Carlo.

NOTE: these functions operate on batches of MCMC configurations and should not
be vmapped.
"""

import chex
from psispin import constants
from psispin import networks
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np

def _log_prob_gaussian(x, mu, sigma):
  """Calculates the log probability of Gaussian with diagonal covariance.

  Args:
    x: Positions. Shape (batch, nelectron, 1, ndim) - as used in mh_update.
    mu: means of Gaussian distribution. Same shape as or broadcastable to x.
    sigma: standard deviation of the distribution. Same shape as or
      broadcastable to x.

  Returns:
    Log probability of Gaussian distribution with shape as required for
    mh_update - (batch, nelectron, 1, 1).
  """
  numer = jnp.sum(-0.5 * ((x - mu)**2) / (sigma**2), axis=[1, 2, 3])
  denom = x.shape[-1] * jnp.sum(jnp.log(sigma), axis=[1, 2, 3])
  return numer - denom

def mh_accept(x1, x2, spins1, spins2, lp_1, lp_2, ratio, key, num_accepts):
  """Given state, proposal, and probabilities, execute MH accept/reject step."""
  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2, x1)
  spins_new = jnp.where(cond[..., None], spins2, spins1)
  lp_new = jnp.where(cond, lp_2, lp_1)
  num_accepts += jnp.sum(cond)
  return x_new, spins_new, key, lp_new, num_accepts

# Random spin swapping function for mh_update 
def swap_spins(key: chex.PRNGKey, spins1, spin_update_prob):
    num_electrons = spins1.shape[0]
    lam = spin_update_prob * num_electrons / 2.0

    # Sample number of swaps from Poisson
    key, subkey = jax.random.split(key)
    m = jax.random.poisson(subkey, lam=lam, shape=())

    def single_swap(i, carry):
        spins, key = carry
        key, subkey1, subkey2 = jax.random.split(key, 3)
        indi = jax.random.randint(subkey1, (), 0, num_electrons)
        indj = jax.random.randint(subkey2, (), 0, num_electrons)

        # Swap elements i and j
        temp = spins[indi]
        # tempspins = jnp.copy(spins)
        spins = spins.at[indi].set(spins[indj])
        spins = spins.at[indj].set(temp)
        # jax.debug.print("i, j, s2 = {}", [indi, indj, tempspins, spins])
        return spins, key

    # Perform m swaps using a lax.fori_loop
    def body_fn(i, carry):
        return single_swap(i, carry)

    spins2, _ = lax.fori_loop(0, m, body_fn, (spins1, key))
    # jax.debug.print("m, s1, s2 = {}", [m, spins1 - spins2])
    return spins2

# Batch version using vmap
def batch_swap_spins(key: chex.PRNGKey, spins1, spin_update_prob):
    batch_size = spins1.shape[0]
    keys = jax.random.split(key, batch_size)
    return jax.vmap(swap_spins, in_axes=(0, 0, None))(keys, spins1, spin_update_prob)

def mh_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    ndim=3,
    blocks=1,
    i=0,
    update_spin = False,
    conserve_sz = True,
    spin_update_prob = 0.1,
):
  """Performs one Metropolis-Hastings step using an all-electron move.

  Args:
    params: Wavefunction parameters.
    f: Callable with signature f(params, x) which returns the log of the
      wavefunction (i.e. the square root of the log probability of x).
    data: Initial MCMC configurations (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal. The move proposal is drawn from
      N(0, stddev^2).
    ndim: dimensionality of system.
    blocks: Ignored.
    i: Ignored.
    allow_spin_flips: whether to allow spin flips during training

  Returns:
    (x, key, lp, num_accepts), where:
      x: Updated MCMC configurations.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.
  """
  del i, blocks  # electron index ignored for all-electron moves
  key, subkey = jax.random.split(key)
  x1 = data.positions
  spins1 = data.spins
  x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)  # proposal
  if update_spin:                                              # adding spin proposal
    if conserve_sz:
      spins2 = batch_swap_spins(key, spins1, spin_update_prob)
    else:
      key, subkey = jax.random.split(key)
      spin_signs = jnp.where(jax.random.bernoulli(subkey, p=spin_update_prob, shape=spins1.shape), -1, 1)
      spins2 = spins1 * spin_signs
  else:
    spins2 = spins1
  lp_2 = 2.0 * f(params, x2, spins2)  # log prob of proposal
  ratio = lp_2 - lp_1
  x_new, spins_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, spins1, spins2, lp_1, lp_2, ratio, key, num_accepts)

  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new, 'spins': spins_new}))
  return new_data, key, lp_new, num_accepts

def mh_update_separate_spin_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts_pos,
    num_accepts_spin,
    stddev=0.02,
    ndim=3,
    blocks=1,
    i=0,
    update_spin = False,
    conserve_sz = True,
    spin_update_prob = 0.01,

):
  """Performs one Metropolis-Hastings step using an all-electron move.

  Args:
    params: Wavefunction parameters.
    f: Callable with signature f(params, x) which returns the log of the
      wavefunction (i.e. the square root of the log probability of x).
    data: Initial MCMC configurations (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal. The move proposal is drawn from
      N(0, stddev^2).
    ndim: dimensionality of system.
    blocks: Ignored.
    i: Ignored.
    allow_spin_flips: whether to allow spin flips during training

  Returns:
    (x, key, lp, num_accepts), where:
      x: Updated MCMC configurations.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.
  """
  del i, blocks  # electron index ignored for all-electron moves
  key, subkey = jax.random.split(key)
  x1          = data.positions
  spins1      = data.spins

  # positions move
  x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)  # proposal
  lp_2  = 2.0 * f(params, x2, spins1)  # log prob of proposal
  ratio = lp_2 - lp_1

  x_new, spins_new, key, lp_new_x, num_accepts_pos = mh_accept(
      x1, x2, spins1, spins1, lp_1, lp_2, ratio, key, num_accepts_pos)

  if update_spin:      # adding spin proposal
    lp_1_s            = lp_new_x  # log probability after pos update and before spin update
    if conserve_sz:
      spins2          = batch_swap_spins(key, spins1, spin_update_prob)
    else:
      key, subkey     = jax.random.split(key)
      spin_signs      = jnp.where(jax.random.bernoulli(subkey, p=spin_update_prob, shape=spins1.shape), -1, 1)
      spins2          = spins1 * spin_signs
    lp_2_s            = 2.0 * f(params, x_new, spins2)  # log prob of proposal
    ratio_spin_update = lp_2_s - lp_1_s

    # overwrite spins_new with accepted spin moves
    x_unchanged, spins_new, key, lp_new, num_accepts_spin = mh_accept(
      x_new, x_new, spins1, spins2, lp_1_s, lp_2_s, ratio_spin_update, key, num_accepts_spin)

  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new, 'spins': spins_new}))
  return new_data, key, lp_new, num_accepts_pos, num_accepts_spin

def make_mcmc_step_separate_spin_update(
                   batch_network,
                   batch_per_device,
                   steps            = 10,
                   ndim             = 3,
                   blocks           = 1,
                   update_spin      = False,
                   spin_update_prob = 0.01,
                   conserve_sz      = True,
                   ):
  """Creates the MCMC step function.

  Args:
    batch_network: function, signature (params, x), which evaluates the log of
      the wavefunction (square root of the log probability distribution) at x
      given params. Inputs and outputs are batched.
    batch_per_device: Batch size per device.
    steps: Number of MCMC moves to attempt in a single call to the MCMC step
      function.
    ndim: Dimensionality of the system (usually 3).
    blocks: Number of blocks to split the updates into. If 1, use all-electron
      moves.

  Returns:
    Callable which performs the set of MCMC steps.
  """
  print("Making mcmc step with separate spin updates.")

  inner_fun     = mh_update_separate_spin_update

  def mcmc_step(params, data, key, width):
    """Performs a set of MCMC steps.

    Args:
      params: parameters to pass to the network.
      data: (batched) MCMC configurations to pass to the network.
      key: RNG state.
      width: standard deviation to use in the move proposal.

    Returns:
      (data, pmove), where data is the updated MCMC configurations, key the
      updated RNG state and pmove the average probability a move was accepted.
    """
    pos = data.positions

    def step_fn(i, x):
      return inner_fun(
          params,
          batch_network,
          *x,
          stddev = width,
          ndim   = ndim,
          blocks = blocks,
          i      = i,
          update_spin      = update_spin,
          spin_update_prob = spin_update_prob,
          conserve_sz      = conserve_sz,
          )

    nsteps  = steps * blocks
    logprob = 2.0 * batch_network(params, pos, data.spins)
    new_data, key, _, num_accepts_pos, num_accepts_spin = lax.fori_loop(
        0, nsteps, step_fn, (data, key, logprob, 0.0, 0.0)
    )

    pmove_pos  = jnp.sum(num_accepts_pos) / (nsteps * batch_per_device)
    pmove_pos  = constants.pmean(pmove_pos)

    pmove_spin = jnp.sum(num_accepts_spin) / (nsteps * batch_per_device)
    pmove_spin = constants.pmean(pmove_spin)

    return new_data, jnp.array([pmove_pos, pmove_spin])

  return mcmc_step

def mh_block_update(                                   # SPIN UPDATES ARE NOT IMPLEMENTED
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    ndim=3,
    blocks=1,
    i=0,
):
  """Performs one Metropolis-Hastings step for a block of electrons.

  Args:
    params: Wavefunction parameters.
    f: Callable with LogFermiNetLike signature which returns the log of the
      wavefunction (i.e. the square root of the log probability of x).
    data: Initial MCMC configuration (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    ndim: dimensionality of system.
    blocks: number of blocks to split electron updates into.
    i: index of block of electrons to move.

  Returns:
    (x, key, lp, num_accepts), where:
      x: MCMC configurations with updated positions.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.
  """
  key, subkey = jax.random.split(key)
  batch_size = data.positions.shape[0]
  nelec = data.positions.shape[1] // ndim
  pad = (blocks - nelec % blocks) % blocks
  x1 = jnp.reshape(
      jnp.pad(data.positions, ((0, 0), (0, pad * ndim))),
      [batch_size, blocks, -1, ndim],
  )
  ii = i % blocks
  x2 = x1.at[:, ii].add(
      stddev * jax.random.normal(subkey, shape=x1[:, ii].shape))
  x2 = jnp.reshape(x2, [batch_size, -1])
  if pad > 0:
    x2 = x2[..., :-pad*ndim]
  # log prob of proposal
  lp_2 = 2.0 * f(params, x2, data.spins)
  ratio = lp_2 - lp_1

  x1 = jnp.reshape(x1, [batch_size, -1])
  if pad > 0:
    x1 = x1[..., :-pad*ndim]
  x_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, lp_1, lp_2, ratio, key, num_accepts)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
  return new_data, key, lp_new, num_accepts

def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   ndim=3,
                   blocks=1,
                   update_spin=False,
                   spin_update_prob = 0.01,
                   conserve_sz=True,
                   ):
  """Creates the MCMC step function.

  Args:
    batch_network: function, signature (params, x), which evaluates the log of
      the wavefunction (square root of the log probability distribution) at x
      given params. Inputs and outputs are batched.
    batch_per_device: Batch size per device.
    steps: Number of MCMC moves to attempt in a single call to the MCMC step
      function.
    ndim: Dimensionality of the system (usually 3).
    blocks: Number of blocks to split the updates into. If 1, use all-electron
      moves.

  Returns:
    Callable which performs the set of MCMC steps.
  """
  inner_fun = mh_block_update if blocks > 1 else mh_update

  def mcmc_step(params, data, key, width):
    """Performs a set of MCMC steps.

    Args:
      params: parameters to pass to the network.
      data: (batched) MCMC configurations to pass to the network.
      key: RNG state.
      width: standard deviation to use in the move proposal.

    Returns:
      (data, pmove), where data is the updated MCMC configurations, key the
      updated RNG state and pmove the average probability a move was accepted.
    """
    pos = data.positions

    def step_fn(i, x):
      return inner_fun(
          params,
          batch_network,
          *x,
          stddev=width,
          ndim=ndim,
          blocks=blocks,
          i=i,
          update_spin=update_spin,
          spin_update_prob=spin_update_prob,
          conserve_sz=conserve_sz,
          )

    nsteps = steps * blocks
    logprob = 2.0 * batch_network(params, pos, data.spins)
    new_data, key, _, num_accepts = lax.fori_loop(
        0, nsteps, step_fn, (data, key, logprob, 0.0)
    )
    pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)
    pmove = constants.pmean(pmove)
    return new_data, pmove

  return mcmc_step

def update_mcmc_width(
    t: int,
    width: jnp.ndarray,
    adapt_frequency: int,
    pmove: jnp.ndarray,
    pmoves: np.ndarray,
    pmove_max: float = 0.55,
    pmove_min: float = 0.5,
) -> tuple[jnp.ndarray, np.ndarray]:
  """Updates the width in MCMC steps.

  Args:
    t: Current step.
    width: Current MCMC width.
    adapt_frequency: The number of iterations after which the update is applied.
    pmove: Acceptance ratio in the last step.
    pmoves: Acceptance ratio over the last N steps, where N is the number of
      steps between MCMC width updates.
    pmove_max: The upper threshold for the range of allowed pmove values
    pmove_min: The lower threshold for the range of allowed pmove values

  Returns:
    width: Updated MCMC width.
    pmoves: Updated `pmoves`.
  """

  t_since_mcmc_update = t % adapt_frequency
  # update `pmoves`; `pmove` should be the same across devices
  pmoves[t_since_mcmc_update] = pmove.reshape(-1)[0].item()
  if t > 0 and t_since_mcmc_update == 0:
    if np.mean(pmoves) > pmove_max:
      width *= 1.1
    elif np.mean(pmoves) < pmove_min:
      width /= 1.1
  return width, pmoves


