# Based on "psiformer.py", modified by Max Geier and Alexander Avdoshkin, MIT, 2026
# 
# Original Copyright of psiformer.py:
# Copyright 2023 DeepMind Technologies Limited.
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

"""Attention-based networks for PsiSpin."""

from typing import Mapping, Optional, Sequence, Tuple, Union

import attr
import chex
from psispin import envelopes

from psispin import network_blocks
from psispin import networks
import jax
import jax.numpy as jnp
import numpy as np


@attr.s(auto_attribs=True, kw_only=True)
class SlaterNetOptions(networks.BaseNetworkOptions):
  """Options controlling the Hartree-Fock part of the network architecture.

  Attributes:
    num_layers: Number of MLP layers representing the single orbitals.
    mlp_dim: Dimension of the perceptron layers
    num_perceptrons_per_layer: number of perceptrons per layer
    use_layer_norm: If true, include a layer norm after each num_perceptrons_per_layer of perceptrons.
  
  Notice: The total number of perceptrons is: num_layers * num_perceptrons_per_layer
      After each layer (including multiple perceptrons), the residual output h^(l-1) from the previous layer is summed for a "memory effect".
      If use_layer_norm = True, after each layer a layer norm is applied, for numerical stability
  """

  num_layers: int = 3
  mlp_dim: int = 256
  num_perceptrons_per_layer: int = 2 
  use_layer_norm: bool = False
  mlp_activation_fct: str = "TANH"

def make_layer_norm() ->...:
  """Implementation of LayerNorm."""

  def init(param_shape: int) -> Mapping[str, jnp.ndarray]:
    params = {}
    params['scale'] = jnp.ones(param_shape)
    params['offset'] = jnp.zeros(param_shape)
    return params

  def apply(params: networks.ParamTree,
            inputs: jnp.ndarray,
            axis: int = -1) -> jnp.ndarray:
    mean = jnp.mean(inputs, axis=axis, keepdims=True)
    variance = jnp.var(inputs, axis=axis, keepdims=True)
    eps = 1e-5
    inv = params['scale'] * jax.lax.rsqrt(variance + eps)
    return inv * (inputs - mean) + params['offset']

  return init, apply

def make_mlp(num_perceptron_per_layer: int = 1) ->...:
  """Construct MLP, with final linear projection to embedding size."""
  if num_perceptron_per_layer == 1:
    def init(key: chex.PRNGKey, mlp_dim: int, embed_dim: int,
            ) -> Sequence[networks.Param]:
      key1, key2 = jax.random.split(key)
      weight = (
          jax.random.normal(key1, shape=(embed_dim, mlp_dim)) /
          jnp.sqrt(float(embed_dim)))
      bias = jax.random.normal(key2, shape=(mlp_dim,))
      params = [{'w': weight, 'b': bias}]
      return params

    def apply(params: Sequence[networks.Param],
              inputs: jnp.ndarray) -> jnp.ndarray:
      return jnp.tanh(jnp.dot(inputs, params[0]['w']) + params[0]['b'] )

    return init, apply
  else:
    def init(key: chex.PRNGKey, mlp_dim: int, embed_dim: int,
            ) -> Sequence[networks.Param]:
      params = []
      dims_one_in = [mlp_dim for cnt in range(num_perceptron_per_layer)]
      dims_one_out = [mlp_dim for cnt in range(num_perceptron_per_layer)]
      dims_one_in[0] = embed_dim
      # dims_one_out[-1] = embed_dim
      for i in range(len(dims_one_in)):
        key, subkey = jax.random.split(key)
        params.append(
            network_blocks.init_linear_layer(
                subkey,
                in_dim=dims_one_in[i],
                out_dim=dims_one_out[i],
                include_bias=True))
      return params

    def apply(params: Sequence[networks.Param],
              inputs: jnp.ndarray) -> jnp.ndarray:
      x = inputs
      for i in range(len(params)):
        x = jnp.tanh(network_blocks.linear_layer(x, **params[i]))
      return x

    return init, apply

def make_mlp(activation_fct_name: str = "TANH") ->...:
  """Construct MLP, with final linear projection to embedding size."""
  
  # Select non-linear activation function
  if activation_fct_name.upper() == "TANH":
    activation_fct = jnp.tanh
  elif activation_fct_name.upper() == "ELU":
    activation_fct = jax.nn.elu
  elif activation_fct_name.upper() == "GELU":
    activation_fct = jax.nn.gelu
  
  # Initialize parameters
  def init(key: chex.PRNGKey, input_dim: int, perceptron_dims: Tuple[int, ...],
           ) -> Sequence[networks.Param]:
    params = []
    for i in range(len(perceptron_dims)):
      if i == 0:
        in_dim = input_dim
        perceptron_dim = perceptron_dims[0]
      else:
        in_dim = perceptron_dims[i - 1]
        perceptron_dim = perceptron_dims[i]

      key, subkey = jax.random.split(key)
      params.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=in_dim,
              out_dim=perceptron_dim,
              include_bias=True))
    return params

  # Apply multilayer perceptron
  def apply(params: Sequence[networks.Param],
            inputs: jnp.ndarray) -> jnp.ndarray:
    x = inputs
    for i in range(len(params)):
      x = activation_fct(network_blocks.linear_layer(x, **params[i]))
    return x

  return init, apply


def make_SlaterNet_block(num_layers: int,
                              mlp_dim: int,
                              num_perceptrons_per_layer: int,
                              use_layer_norm: bool = False,
                              mlp_activation_fct: str = "TANH") ->...:
  if use_layer_norm:
    layer_norm_init, layer_norm_apply = make_layer_norm()

  mlp_init, mlp_apply = make_mlp(mlp_activation_fct)

  def init(key: chex.PRNGKey) -> networks.ParamTree: # replaced qkv_dim -> mlp_dim because attention is skipped so only mlp is present
    params = {}
    ln_params = []
    mlp_params = []

    for _ in range(num_layers):
      key, mlp_key = jax.random.split(key, 2)
      if use_layer_norm:
        ln_params.append([layer_norm_init(mlp_dim)])
      mlp_params.append(mlp_init(mlp_key, input_dim = mlp_dim, perceptron_dims = (mlp_dim, ) * num_perceptrons_per_layer))

    params['ln'] = ln_params
    params['mlp'] = mlp_params

    return params

  def apply(params: networks.ParamTree, single_particle_stream: jnp.ndarray) -> jnp.ndarray:
    x = single_particle_stream
    for layer in range(num_layers):
      # MLP
      assert isinstance(params['mlp'][layer], (tuple, list))
      mlp_output = mlp_apply(params['mlp'][layer], x)

      # Residual + optional LayerNorm.
      x = x + mlp_output
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][0], x)

    return x

  return init, apply


def make_SlaterNet_layers(
    nspins: Tuple[int, ...],
    options: SlaterNetOptions,
) -> Tuple[networks.InitLayersFn, networks.ApplyLayersFn]:
  """Creates the permutation-equivariant layers for SlaterNet.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    options: network options.

  Returns:
    Tuple of init, apply functions.
  """
  del nspins  # Unused.

  SlaterNet_init, SlaterNet_apply = make_SlaterNet_block(
      num_layers=options.num_layers,
      mlp_dim=options.mlp_dim,
      num_perceptrons_per_layer=options.num_perceptrons_per_layer,
      use_layer_norm=options.use_layer_norm,
      mlp_activation_fct=options.mlp_activation_fct
  )

  def init(key: chex.PRNGKey) -> Tuple[int, networks.ParamTree]:
    """Returns tuple of output dimension from the final layer and parameters."""
    params = {}
    key, subkey = jax.random.split(key)
    feature_dims, params['input'] = options.feature_layer.init()
    one_electron_feature_dim, _ = feature_dims
    # Concatenate spin of each electron with other one-electron features.
    feature_dim = one_electron_feature_dim + 1

    # Map to MLP dim.
    key, subkey = jax.random.split(key)
    params['embed'] = network_blocks.init_linear_layer(
        subkey, in_dim=feature_dim, out_dim=options.mlp_dim, include_bias=False
    )['w']

    # MLP block params.
    key, subkey = jax.random.split(key)
    params.update(SlaterNet_init(key))

    return options.mlp_dim, params

  def apply(
      params,
      *,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      spins: jnp.ndarray,
  ) -> jnp.ndarray:
    """Applies the SlaterNet interaction layers to a walker configuration.

    Args:
      params: parameters for the interaction and permuation-equivariant layers.
      ae: one-electron position features.
      r_ae: one-electron position norms.
      ee: electron-electron vectors.
      r_ee: electron-electron distances.
      spins: spin of each electron.

    Returns:
      Array of shape (nelectron, output_dim), where the output dimension,
      output_dim, is given by init, and is suitable for projection into orbital
      space.
    """
    # Only one-electron features are used by SlaterNet.
    ae_features, _ = options.feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input']
    )

    # For Hartree-Fock, the spin feature is required for correct permutation equivariance, as in PsiFormer.
    ae_features = jnp.concatenate((ae_features, spins[..., None]), axis=-1)

    features = ae_features  # Just 1-electron stream for now.

    # Embed into attention dimension.
    x = jnp.dot(features, params['embed'])

    return SlaterNet_apply(params, x)

  return init, apply


def make_fermi_net(
    nspins: Tuple[int, ...],
    *,
    ndim: int = 3,
    determinants: int = 1,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # SlaterNet-specific kwargs below.
    num_layers: int,
    mlp_dim: int,
    num_perceptrons_per_layer: int,
    use_layer_norm: bool,
    pbc_lattice: jnp.ndarray,
    use_spin_channels: bool = True,
    mlp_activation_fct: str,
) -> networks.Network:
  """SlaterNet implementation of Hartree Fock.

  Includes standard envelope and determinants.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    ndim: Dimension of the system. Change only with caution.
    determinants: Number of determinants.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    complex_output: If true, the wavefunction output is complex-valued.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    mlp_dim: dimension of the perceptron layers
    num_perceptrons_per_layer: number of perceptrons between residual addition and layer norm
    use_layer_norm: If true, use layer_norm on MLP.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network.
  """

  if not envelope:
    envelope = envelopes.make_null_envelope()

  if not feature_layer:
    feature_layer = networks.make_ferminet_features(
        nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  options = SlaterNetOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,  # Required for SlaterNet, taken over from Psiformer.
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      mlp_dim=mlp_dim,
      num_perceptrons_per_layer=num_perceptrons_per_layer,
      use_layer_norm=use_layer_norm,
      mlp_activation_fct=mlp_activation_fct,
      pbc_lattice = pbc_lattice,
  )  # pytype: disable=wrong-keyword-args

  SlaterNet_layers = make_SlaterNet_layers(nspins, options)

  orbitals_init, orbitals_apply = networks.make_orbitals(
      nspins=nspins,
      options=options,
      equivariant_layers=SlaterNet_layers,
      use_spin_channels=use_spin_channels,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of SlaterNet.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute value of the network evaluated at x.
    """
    orbitals = orbitals_apply(params, pos, spins)
    if options.states:
      batch_logdet_matmul = jax.vmap(network_blocks.logdet_matmul, in_axes=0)
      orbitals = [
          jnp.reshape(orbital, (options.states, -1) + orbital.shape[1:])
          for orbital in orbitals
      ]
      result = batch_logdet_matmul(orbitals)
    else:
      result = network_blocks.logdet_matmul(orbitals)
    if 'state_scale' in params:
      # only used at inference time for excited states
      result = result[0], result[1] + params['state_scale']
    return result

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )
