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
from psispin import networks
import jax
import jax.numpy as jnp
import numpy as np
from jax.debug import print as jprint

from jax import lax
from psispin.utils import utils
import folx

def make_zeeman(
    signed_network: networks.FermiNetLike,
    zeeman_field = None,
):
  """Returns the electron-electron potential.

  
  Args:
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
  """

  def _field(position: jnp.ndarray) -> jnp.ndarray:
    return [0,0,0]

  if zeeman_field == None:
      zeeman_field = _field

  def _zeeman_over_f(params, data):

    n_electrons = data.spins.shape[0]
    ndim = data.positions.shape[0] // n_electrons
    
    sign_psi, log_psi = signed_network(params, data.positions, data.spins)

    # Reshape positions into an array of shape (n_electrons, ndim)
    positions = jnp.reshape(data.positions, (n_electrons, ndim))

    # We need to sum over all positions

    def _inner(i, val):
      # Flip the i-th electron's spin. Here we assume spins are represented as -1 or 1.
      new_spins = data.spins.at[i].set(-1 * data.spins[i])

      sign_flip, log_flip = signed_network(params, data.positions, new_spins)
      # Contribution from electron i: 0.5 * (psi_flip / psi)
      field_val = zeeman_field(positions[i])      # taking the components of the Zeeman field

      psi_ratio = jnp.exp(1.j*(sign_flip - sign_psi) + log_flip-log_psi)

      x_contribution = psi_ratio*field_val[0]
      y_contribution = -1.0j * data.spins[i] * psi_ratio*field_val[1]
      z_contribution = data.spins[i] * field_val[2]

      return val + x_contribution + y_contribution + z_contribution
    

    energy = jax.lax.fori_loop(0, n_electrons, _inner, 0.0)

    return energy

  return _zeeman_over_f



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

  zeeman_field = potential_kwargs['zeeman_field']
  ze = make_zeeman(f, zeeman_field=zeeman_field)

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
    #potential = potential_energy(ae, ee)    
    kinetic = ke(params, data)
    zeeman_term = ze(params, data)
    return kinetic + zeeman_term, None
  return _e_l
