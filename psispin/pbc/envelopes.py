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

"""Multiplicative envelopes appropriate for periodic boundary conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

import itertools
from typing import Mapping, Optional, Sequence, Tuple, Union

from psispin import envelopes

import jax.numpy as jnp
import numpy as np

from jax.debug import print as jprint


def make_identity_envelope() -> envelopes.Envelope:
  """ Return envelopes that implement twisted periodic boundary conditions.
  For periodic boundary conditions, the envelope is identity.
  """
  jprint("pbc.envelopes: using identity envelope")
  def init(
      output_dims: Sequence[int], ndim: int = 3
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    """See psispin.envelopes.EnvelopeInit."""
    del output_dims, ndim  # unused
    params = [{}]
    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray) -> float:
    return 1.0

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)

def make_twisted_bc_envelope(lattice: jnp.ndarray, twist_angles: jnp.array = jnp.array([0.0, 0.0])
                             ) -> envelopes.Envelope:
  """ Return envelopes that implement twisted periodic boundary conditions. """
  print("Using twisted PBC envelope.")

  def init(
      output_dims: Sequence[int], ndim: int = 3
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    """See psispin.envelopes.EnvelopeInit."""
    del output_dims, ndim  # unused
    params = [{}]
    return params

  rec = jnp.linalg.inv(lattice) # Dividing out a factor of 2pi here,
                                # because (i) twist angles in rad contain the factor 2pi,
                                #  and (ii) in phase_ae it would be divided out, too
  G1 = rec[0]
  G2 = rec[1]
  twist_vector =  G1 * twist_angles[0] + G2 * twist_angles[1]

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray) -> jnp.ndarray:
    # Use positions mapped to first unit cell for envelope calculation
    phase_ae = jnp.einsum('il,jl->ji', rec, ae) # using rec / (2 pi) ad defined above
    phase_prim_ae = (phase_ae + 0.5)  % 1 - 0.5
    prim_ae = jnp.einsum('il,jl->ji', lattice, phase_prim_ae)
    return jnp.exp(1j * jnp.sum(prim_ae * twist_vector, axis=-1, keepdims=True))

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)

def make_multiwave_envelope(kpoints: jnp.ndarray) -> envelopes.Envelope:
  """Returns an oscillatory envelope.

  Envelope consists of a sum of truncated 3D Fourier series, one centered on
  each atom, with Fourier frequencies given by kpoints:

    sigma_{2i}*cos(kpoints_i.r_{ae}) + sigma_{2i+1}*sin(kpoints_i.r_{ae})

  Initialization sets the coefficient of the first term in each
  series to 1, and all other coefficients to 0. This corresponds to the
  cosine of the first entry in kpoints. If this is [0, 0, 0], the envelope
  will evaluate to unity at the beginning of training.

  Args:
    kpoints: Reciprocal lattice vectors of terms included in the Fourier
      series. Shape (nkpoints, ndim) (Note that ndim=3 is currently
      a hard-coded default).

  Returns:
    An instance of psispin.envelopes.Envelope with apply_type
    envelopes.EnvelopeType.PRE_DETERMINANT
  """

  jprint("pbc.envelopes: using multiwave envelope")
  def init(
      output_dims: Sequence[int], ndim: int = 3
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    """See psispin.envelopes.EnvelopeInit."""
    del ndim  # unused
    params = []
    nk = kpoints.shape[0]
    for output_dim in output_dims:
      params.append({'sigma': jnp.zeros((2 * nk, output_dim))})
      params[-1]['sigma'] = params[-1]['sigma'].at[0, :].set(1.0)
    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            sigma: jnp.ndarray) -> jnp.ndarray:
    """See psispin.envelopes.EnvelopeApply."""
    del r_ae, r_ee  # unused
    phase_coords = ae @ kpoints.T
    waves = jnp.concatenate((jnp.cos(phase_coords), jnp.sin(phase_coords)),
                            axis=1)
    env = waves @ (sigma**2.0)
    return env

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


def make_kpoints(
    lattice: Union[np.ndarray, jnp.ndarray],
    spins: Tuple[int, int],
    min_kpoints: Optional[int] = None,
) -> jnp.ndarray:
  """Generates an array of reciprocal lattice vectors.

  Args:
    lattice: Matrix whose columns are the primitive lattice vectors of the
      system, shape (ndim, ndim). (Note that ndim=3 is currently
      a hard-coded default).
    spins: Tuple of the number of spin-up and spin-down electrons.
    min_kpoints: If specified, the number of kpoints which must be included in
      the output. The number of kpoints returned will be the
      first filled shell which is larger than this value. Defaults to None,
      which results in min_kpoints == sum(spins).

  Raises:
    ValueError: Fewer kpoints requested by min_kpoints than number of
      electrons in the system.

  Returns:
    jnp.ndarray, shape (nkpoints, ndim), an array of reciprocal lattice
      vectors sorted in ascending order according to length.
  """
  rec_lattice = 2 * jnp.pi * jnp.linalg.inv(lattice)
  # Calculate required no. of k points
  if min_kpoints is None:
    min_kpoints = sum(spins)
  elif min_kpoints < sum(spins):
    raise ValueError(
        'Number of kpoints must be equal or greater than number of electrons')

  dk = 1 + 1e-5
  # Generate ordinals of the lowest min_kpoints kpoints
  max_k = int(jnp.ceil(min_kpoints * dk)**0.5) # MG: Adjust for 2D
  ordinals = sorted(range(-max_k, max_k+1), key=abs)
  ordinals = jnp.asarray(list(itertools.product(ordinals, repeat=2))) # MG: Adjust for 2D

  kpoints = ordinals @ rec_lattice.T
  # kpoints = ordinals @ rec_lattice
  kpoints = jnp.asarray(sorted(kpoints, key=jnp.linalg.norm))
  k_norms = jnp.linalg.norm(kpoints, axis=1)

  return kpoints[k_norms <= k_norms[min_kpoints - 1] * dk]
