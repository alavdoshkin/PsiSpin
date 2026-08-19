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

"""Evaluating the Hamiltonian on a wavefunction."""

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import chex
from psispin import networks
from psispin.utils import utils
import folx
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Protocol

Array = Union[jnp.ndarray, np.ndarray]


class LocalEnergy(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Returns the local energy of a Hamiltonian at a configuration.

    Args:
      params: network parameters.
      key: JAX PRNG state.
      data: MCMC configuration to evaluate.
    """


class MakeLocalEnergy(Protocol):

  def __call__(
      self,
      f: networks.FermiNetLike,
      nspins: Sequence[int],
      use_scan: bool = False,
      complex_output: bool = False,
      **kwargs: Any
  ) -> LocalEnergy:
    """Builds the LocalEnergy function.

    Args:
      f: Callable which evaluates the sign and log of the magnitude of the
        wavefunction.
      nspins: Number of particles of each spin.
      use_scan: Whether to use a `lax.scan` for computing the laplacian.
      complex_output: If true, the output of f is complex-valued.
      **kwargs: additional kwargs to use for creating the specific Hamiltonian.
    """


KineticEnergy = Callable[
    [networks.ParamTree, networks.FermiNetData], jnp.ndarray
]


def local_kinetic_energy(
    f: networks.FermiNetLike,
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = 'default',
) -> KineticEnergy:
  r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

  Args:
    f: Callable which evaluates the wavefunction as a
      (sign or phase, log magnitude) tuple.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.
    laplacian_method: Laplacian calculation method. One of:
      'default': take jvp(grad), looping over inputs
      'folx': use Microsoft's implementation of forward laplacian

  Returns:
    Callable which evaluates the local kinetic energy,
    -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
  """

  phase_f = utils.select_output(f, 0)
  logabs_f = utils.select_output(f, 1)

  if laplacian_method == 'default':
    def _lapl_over_f(params, data):
      n = data.positions.shape[0]
      eye = jnp.eye(n)
      grad_f = jax.grad(logabs_f, argnums=1)
      def grad_f_closure(x):
        return grad_f(params, x, data.spins)

      primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)

      if complex_output:
        grad_phase = jax.grad(phase_f, argnums=1)
        def grad_phase_closure(x):
          return grad_phase(params, x, data.spins)
        phase_primal, dgrad_phase = jax.linearize(
            grad_phase_closure, data.positions)
        hessian_diagonal = (
            lambda i: dgrad_f(eye[i])[i] + 1.j * dgrad_phase(eye[i])[i]
        )
      else:
        hessian_diagonal = lambda i: dgrad_f(eye[i])[i]

      if use_scan:
        _, diagonal = lax.scan(
            lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=n)
        result = -0.5 * jnp.sum(diagonal)
      else:
        result = -0.5 * lax.fori_loop(
            0, n, lambda i, val: val + hessian_diagonal(i), 0.0)
      result -= 0.5 * jnp.sum(primal ** 2)
      if complex_output:
        result += 0.5 * jnp.sum(phase_primal ** 2)
        result -= 1.j * jnp.sum(primal * phase_primal)
      return result

  elif laplacian_method == 'folx':
    def _lapl_over_f(params, data):
      f_closure = lambda x: f(params, x, data.spins)
      f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=6)
      output = f_wrapped(data.positions)
      result = - (output[1].laplacian +
                  jnp.sum(output[1].jacobian.dense_array ** 2)) / 2
      if complex_output:
        result -= 0.5j * output[0].laplacian
        result += 0.5 * jnp.sum(output[0].jacobian.dense_array ** 2)
        result -= 1.j * jnp.sum(output[0].jacobian.dense_array *
                                output[1].jacobian.dense_array)
      return result

  else:
    raise NotImplementedError(f'Laplacian method {laplacian_method} '
                              'not implemented.')

  return _lapl_over_f


def excited_kinetic_energy_matrix(
    f: networks.FermiNetLike,
    states: int,
    complex_output: bool = False,
    laplacian_method: str = 'default') -> KineticEnergy:
  """Creates a f'n which evaluates the matrix of local kinetic energies.

  Args:
    f: A network which returns a tuple of sign(psi) and log(|psi|) arrays, where
      each array contains one element per excited state.
    states: the number of excited states
    complex_output: If true, the output of f is complex-valued.
    laplacian_method: Laplacian calculation method. One of:
      'default': take jvp(grad), looping over inputs
      'folx': use Microsoft's implementation of forward laplacian

  Returns:
    A function which computes the matrices (psi) and (K psi), which are the
      value of the wavefunction and the kinetic energy applied to the
      wavefunction for all combinations of electron sets and excited states.
  """

  def _lapl_all_states(params, pos, spins):
    """Return K psi/psi for each excited state."""
    n = pos.shape[0]
    eye = jnp.eye(n)
    grad_f = jax.jacrev(utils.select_output(f, 1), argnums=1)
    grad_f_closure = lambda x: grad_f(params, x, spins)
    primal, dgrad_f = jax.linearize(grad_f_closure, pos)

    if complex_output:
      grad_phase = jax.jacrev(utils.select_output(f, 0), argnums=1)
      def grad_phase_closure(x):
        return grad_phase(params, x, spins)
      phase_primal, dgrad_phase = jax.linearize(grad_phase_closure, pos)
      hessian_diagonal = (
          lambda i: dgrad_f(eye[i])[:, i] + 1.j * dgrad_phase(eye[i])[:, i]
      )
    else:
      phase_primal = 1.0
      hessian_diagonal = lambda i: dgrad_f(eye[i])[:, i]

    if complex_output:
      if pos.dtype == jnp.float32:
        dtype = jnp.complex64
      elif pos.dtype == jnp.float64:
        dtype = jnp.complex128
      else:
        raise ValueError(f'Unsupported dtype for input: {pos.dtype}')
    else:
      dtype = pos.dtype

    result = -0.5 * lax.fori_loop(
        0, n, lambda i, val: val + hessian_diagonal(i),
        jnp.zeros(states, dtype=dtype))
    result -= 0.5 * jnp.sum(primal ** 2, axis=-1)
    if complex_output:
      result += 0.5 * jnp.sum(phase_primal ** 2, axis=-1)
      result -= 1.j * jnp.sum(primal * phase_primal, axis=-1)

    return result

  def _lapl_over_f(params, data):
    """Return the kinetic energy (divided by psi) summed over excited states."""
    pos_ = jnp.reshape(data.positions, [states, -1])
    spins_ = jnp.reshape(data.spins, [states, -1])

    if laplacian_method == 'default':
      vmap_f = jax.vmap(f, (None, 0, 0))
      sign_mat, log_mat = vmap_f(params, pos_, spins_)
      vmap_lapl = jax.vmap(_lapl_all_states, (None, 0, 0))
      lapl = vmap_lapl(params, pos_, spins_)  # K psi_i(r_j) / psi_i(r_j)
    elif laplacian_method == 'folx':
      # CAUTION!! Only the first array of spins is being passed!
      f_closure = lambda x: f(params, x, spins_[0])
      f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=6)
      sign_out, log_out = folx.batched_vmap(f_wrapped, 1)(pos_)
      log_mat = log_out.x
      lapl = -(log_out.laplacian +
               jnp.sum(log_out.jacobian.dense_array ** 2, axis=-2)) / 2
      if complex_output:
        sign_mat = sign_out.x
        lapl -= 0.5j * sign_out.laplacian
        lapl += 0.5 * jnp.sum(sign_out.jacobian.dense_array ** 2, axis=-2)
        lapl -= 1.j * jnp.sum(sign_out.jacobian.dense_array *
                              log_out.jacobian.dense_array, axis=-2)
      else:
        sign_mat = sign_out
    else:
      raise NotImplementedError(f'Laplacian method {laplacian_method} '
                                'not implemented with excited states.')

    # psi_i(r_j)
    # subtract off largest value to avoid under/overflow
    if complex_output:
      psi_mat = jnp.exp(log_mat + 1.j * sign_mat - jnp.max(log_mat))
    else:
      psi_mat = sign_mat * jnp.exp(log_mat - jnp.max(log_mat))
    kpsi_mat = lapl * psi_mat  # K psi_i(r_j)
    return psi_mat, kpsi_mat

  return _lapl_over_f


def potential_electron_electron(r_ee: Array) -> jnp.ndarray:
  """Returns the electron-electron potential.

  Args:
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
  """
  r_ee = r_ee[jnp.triu_indices_from(r_ee[..., 0], 1)]
  return (1.0 / r_ee).sum()


