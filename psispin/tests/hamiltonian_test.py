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

"""Tests for psispin.hamiltonian."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from psispin import base_config
from psispin import hamiltonian
from psispin import networks
import jax
import jax.numpy as jnp
import numpy as np


def h_atom_log_psi(param, xs, spins):
  del param, spins
  # log of exact hydrogen wavefunction.
  return -jnp.abs(jnp.linalg.norm(xs))


def h_atom_log_psi_signed(param, xs, spins):
  log_psi = h_atom_log_psi(param, xs, spins)
  return jnp.ones_like(log_psi), log_psi


def kinetic_from_hessian(log_f):

  def kinetic_operator(params, pos, spins):
    f = lambda x: jnp.exp(log_f(params, x, spins))
    ys = f(pos)
    hess = jax.hessian(f)(pos)
    return -0.5 * jnp.trace(hess) / ys

  return kinetic_operator


def kinetic_from_hessian_log(log_f):

  def kinetic_operator(params, pos, spins):
    f = lambda x: log_f(params, x, spins)
    grad_f = jax.grad(f)(pos)
    hess = jax.hessian(f)(pos)
    return -0.5 * (jnp.trace(hess)  + jnp.sum(grad_f**2))

  return kinetic_operator


class HamiltonianTest(parameterized.TestCase):

  @parameterized.parameters(['default', 'folx'])
  def test_local_kinetic_energy(self, laplacian):

    dummy_params = {}
    xs = np.random.normal(size=(3,))
    spins = np.ones(shape=(1,))
    expected_kinetic_energy = -(1 - 2 / np.abs(np.linalg.norm(xs))) / 2

    kinetic = hamiltonian.local_kinetic_energy(h_atom_log_psi_signed,
                                               laplacian_method=laplacian)
    kinetic_energy = kinetic(
        dummy_params,
        networks.FermiNetData(positions=xs, spins=spins),
    )
    np.testing.assert_allclose(
        kinetic_energy, expected_kinetic_energy, rtol=1.e-5)


class LaplacianTest(parameterized.TestCase):

  @parameterized.parameters(['default', 'folx'])
  def test_laplacian(self, laplacian):

    xs = np.random.uniform(size=(100, 3))
    spins = np.ones(shape=(1,))
    data = networks.FermiNetData(positions=xs, spins=spins)
    dummy_params = {}
    t_l_fn = jax.vmap(
        hamiltonian.local_kinetic_energy(h_atom_log_psi_signed,
                                         laplacian_method=laplacian),
        in_axes=(
            None,
            networks.FermiNetData(positions=0, spins=None),
        ),
    )
    t_l = t_l_fn(dummy_params, data)
    hess_t = jax.vmap(
        kinetic_from_hessian(h_atom_log_psi),
        in_axes=(None, 0, None),
    )(dummy_params, xs, spins)
    np.testing.assert_allclose(t_l, hess_t, rtol=1E-5)

  @parameterized.parameters(
      itertools.product([True, False], ['default', 'folx'])
  )
  def test_fermi_net_laplacian(self, full_det, laplacian):
    np.random.seed(12)
    nspins = (2, 3)
    batch = 4
    cfg = base_config.default()
    cfg.network.full_det = full_det
    cfg.network.ferminet.hidden_dims = ((8, 4),) * 2
    cfg.network.determinants = 2
    feature_layer = networks.make_ferminet_features(
        cfg.system.electrons,
        cfg.system.ndim,
    )
    network = networks.make_fermi_net(
        nspins,
        full_det=full_det,
        feature_layer=feature_layer,
        pbc_lattice=jnp.eye(3),
        **cfg.network.ferminet
    )
    log_network = lambda *args, **kwargs: network.apply(*args, **kwargs)[1]
    key = jax.random.PRNGKey(47)
    params = network.init(key)
    xs = np.random.normal(scale=5, size=(batch, sum(nspins) * 3))
    spins = np.sign(np.random.normal(scale=1, size=(batch, sum(nspins))))
    t_l_fn = jax.jit(
        jax.vmap(
            hamiltonian.local_kinetic_energy(network.apply,
                                             laplacian_method=laplacian),
            in_axes=(
                None,
                networks.FermiNetData(positions=0, spins=0),
            ),
        )
    )
    t_l = t_l_fn(
        params,
        networks.FermiNetData(positions=xs, spins=spins),
    )
    hess_t_fn = jax.jit(
        jax.vmap(
            kinetic_from_hessian_log(log_network),
            in_axes=(None, 0, 0),
        )
    )
    hess_t = hess_t_fn(params, xs, spins)
    if hess_t.dtype == jnp.float64:
      atol, rtol = 1.e-10, 1.e-10
    else:
      # This needs a low tolerance because on fast math optimization in CPU can
      # substantially affect floating point expressions. See
      # https://github.com/jax-ml/jax/issues/6566.
      atol, rtol = 4.e-3, 4.e-3
    np.testing.assert_allclose(t_l, hess_t, atol=atol, rtol=rtol)


if __name__ == '__main__':
  absltest.main()
