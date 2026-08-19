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

"""
Contains:


"""

import itertools
from typing import Callable, Optional, Sequence, Tuple

import chex
from psispin import hamiltonian
from psispin.pbc import hamiltonian as pbc_hamiltonian

from psispin import networks
import jax
import jax.numpy as jnp
import numpy as np
from jax.debug import print as jprint

from jax import lax
from psispin.utils import utils
import folx


def make_soc(
    f: networks.FermiNetLike,
    kappa = None,
):
  r"""Creates a function to for the spin-orbit coupling term.

  Args:
    f: Callable which evaluates the wavefunction as a
      (sign or phase, log magnitude) tuple.
    kappa: 3x3 array defining the SOC strength and type.
  Returns:
    Callable which evaluates soc term
  """

  #if kappa is None:
  #  kappa = np.eye(2)

  phase_f = utils.select_output(f, 0)
  logabs_f = utils.select_output(f, 1)

  def _soc_over_f(params, data):

      #n = data.positions.shape[0]
      #eye = jnp.eye(n)
      grad_f = jax.grad(logabs_f, argnums=1)
      primal = grad_f(params, data.positions, data.spins)
      n_electrons = data.spins.shape[0]
      ndim = data.positions.shape[0] // n_electrons


      grad_phase = jax.grad(phase_f, argnums=1)
      phase_primal = grad_phase(params, data.positions, data.spins)
      phase_psi, log_psi = f(params, data.positions, data.spins)
      grads = jnp.reshape(primal + 1.j*phase_primal, (n_electrons, ndim))


      def _inner(i, val):
          # Flip the i-th electron's spin. Here we assume spins are represented as -1 or 1.
          new_spins = data.spins.at[i].set(-1 * data.spins[i])

          grad_psi = grads[i]
          phase_flip, log_flip = f(params, data.positions, new_spins)
          psi_ratio = jnp.exp(1.j*(phase_flip - phase_psi)+log_flip-log_psi)     # equal to psi_flip / psi

          # the SOC term is kappa_{ij} pauli_i (i log grad_j) (psi(s')/psi(s))

          flipped_abs_grad = grad_f(params, data.positions, new_spins)
          flipped_phase_grad = grad_phase(params, data.positions, new_spins)
          flipped_grad_psi = jnp.reshape(flipped_abs_grad+1.j*flipped_phase_grad, (n_electrons, ndim))[i]

          pauli_x_contrib = (-1.j*flipped_grad_psi @ kappa[0]) * psi_ratio
          pauli_y_contrib = -1.0j*data.spins[i]*(-1.j*flipped_grad_psi @ kappa[1]) * psi_ratio
          pauli_z_contrib = data.spins[i]*(-1.j*grad_psi @ kappa[2])

          return val + pauli_x_contrib + pauli_y_contrib + pauli_z_contrib
            
      # Compute the SOC term    
      result = jax.lax.fori_loop(0, n_electrons, _inner, 0.0)
      return result
        
  return _soc_over_f



def local_energy(
    f: networks.FermiNetLike,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
    states: int = 0,
    lattice: Optional[jnp.ndarray] = None,
    heg: bool = True,
    convergence_radius: int = 5,
    potential_type = 'Coulomb',
    potential_kwargs = {}
) -> hamiltonian.LocalEnergy:
  """Creates the local energy function in periodic boundary conditions.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.
    states: Number of excited states to compute. Not implemented, only present
      for consistency of calling convention.
    lattice: Shape (ndim, ndim). Matrix of lattice vectors. Default: identity
      matrix.
    heg: bool. Flag to enable features specific to the electron gas.
    convergence_radius: int. Radius of cluster summed over by Ewald sums.

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  print("Using customized local_energy from pbc.hamiltonian ")
  if states:
    raise NotImplementedError('Excited states not implemented with PBC.')
  del nspins
  assert lattice is not None, "pbc.hamiltonian.local_energy requires lattice to be passed"

  if potential_type == 'Coulomb':
      coloumb_strength = potential_kwargs['coloumb_strength']
      potential_energy = pbc_hamiltonian.make_2DCoulomb_potential(
      lattice, convergence_radius, coloumb_strength
    )

  if potential_kwargs['laplacian_method'] in {'default'}:
    print("local_energy, using laplacian_method: " + potential_kwargs['laplacian_method'])
    ke = hamiltonian.local_kinetic_energy(f, use_scan=use_scan,
                                          complex_output=complex_output,
                                          laplacian_method=potential_kwargs['laplacian_method'])
    
  elif potential_kwargs['laplacian_method'] in {'fwdlap', 'folx'}:
    if complex_output:
      def _lapl_over_f(params, data):
        f_closure = lambda x: f(params, x, data.spins)
        f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=6)
        output = f_wrapped(data.positions)
        result = - (output[1].laplacian +
                    jnp.sum(output[1].jacobian.dense_array ** 2)) / 2
        result -= 0.5j * output[0].laplacian
        result += 0.5 * jnp.sum(output[0].jacobian.dense_array ** 2)
        result -= 1.j * jnp.sum(output[0].jacobian.dense_array *
                                output[1].jacobian.dense_array)
        return jnp.real(result)
    else:
      def _lapl_over_f(params, data):
        f_closure = lambda x: f(params, x, data.spins)
        f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=6)
        output = f_wrapped(data.positions)
        result = - (output[1].laplacian +
                    jnp.sum(output[1].jacobian.dense_array ** 2)) / 2
        return result
      
    print("local_energy, using fwdlap from folx with complex_output: " + str(complex_output))
    ke = _lapl_over_f

  if_soc = False
  if 'kappa_soc' in potential_kwargs:
    kappa = potential_kwargs['kappa_soc']
    soc = make_soc(f, kappa = kappa)
    if_soc = 'True'

  def _e_l(
        params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Returns the total energy.

      Args:
        params: network parameters.
        key: RNG state.
        data: MCMC configuration.
    """
    del key  # unused
    ae, ee, _, _ = networks.construct_input_features(
          data.positions, ndim=2)
      
    kinetic = ke(params, data)
    energy = kinetic
    energy += potential_energy(ae, ee)
    
    if if_soc:
        energy += soc(params, data)
    
          
    
    return energy, None
  return _e_l
