# Copyright 2023 DeepMind Technologies Limited.
# Modifications Copyright (c) 2026 Max Geier, Alexander Avdoshkin, Massachusetts Institute of Technology, MA, USA
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
class PsiformerOptions(networks.BaseNetworkOptions):
  """Options controlling the Psiformer part of the network architecture.

  Attributes:
    num_layers: Number of self-attention layers.
    num_heads: Number of multihead self-attention heads.
    attn_dim: Embedding dimension for each self-attention head.
    value_dim: Dimension of the value vector.
    mlp_dim: Tuple of sizes of hidden dimension of the MLP. Note that
      this does not include the final projection to the embedding dimension.
    num_perceptrons_per_layer: Number of perceptron layers after each self-attention layer.
    use_layer_norm: If true, include a layer norm on both attention and MLP.
    mlp_activation_fct: Non-linear activation function used in the perceptron layers
  """

  num_layers: int = 2
  num_heads: int = 4
  attn_dim: int = 64
  value_dim: int = 64
  mlp_dim: int = 256,
  num_perceptrons_per_layer: int = 1,
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

def make_multi_head_attention(num_heads: int, attn_dim: int, value_dim: int, mlp_dim: int) ->...:
  """FermiNet-style version of latent MultiHeadAttention."""

  # Linear layer plus reshape final dimensions to num_heads, attn_dim.
  def linear_projection_qk(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    y = jnp.dot(x, weights)
    return y.reshape(*x.shape[:-1], num_heads, attn_dim)
  
  def linear_projection_values(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    y = jnp.dot(x, weights)
    return y.reshape(*x.shape[:-1], num_heads, value_dim)

  def init(key: chex.PRNGKey,
           ) -> Mapping[str, jnp.ndarray]:

    key, *subkeys = jax.random.split(key, num=5)
    params = {}
    params['q_w'] = network_blocks.init_linear_layer(
        subkeys[0], in_dim=mlp_dim, out_dim=num_heads * attn_dim, include_bias=False)['w']
    params['k_w'] = network_blocks.init_linear_layer(
        subkeys[1], in_dim=mlp_dim, out_dim=num_heads * attn_dim, include_bias=False)['w']
    params['v_w'] = network_blocks.init_linear_layer(
        subkeys[2], in_dim=mlp_dim, out_dim=num_heads * value_dim, include_bias=False)['w']
    params['attn_output'] = network_blocks.init_linear_layer(
        subkeys[3], in_dim=num_heads * value_dim, out_dim=mlp_dim,
        include_bias=False)['w']

    return params

  def apply(params: networks.ParamTree, single_electron_stream: jnp.ndarray) -> jnp.ndarray:
    """Computes MultiHeadAttention with keys, queries and values.

    Args:
      params: Parameters for attention embeddings.
      query: Shape [..., q_index_dim, q_d]
      key: Shape [..., kv_index_dim, q_d]
      value: Shape [..., kv_index_dim, kv_d]

    Returns:
      A projection of attention-weighted value projections.
      Shape [..., q_index_dim, output_channels]
    """

    # Projections for q, k, v.
    # Output shape: [..., index_dim, num_heads, attn_dim].
    q = linear_projection_qk(single_electron_stream, params['q_w'])
    k = linear_projection_qk(single_electron_stream, params['k_w'])
    v = linear_projection_values(single_electron_stream, params['v_w'])

    attn_logits = jnp.einsum('...thd,...Thd->...htT', k, q) # for reciprocal relations, use same matrix to generate keys and values
    scale = 1. / np.sqrt(attn_dim)
    attn_logits *= scale

    attn_weights = jax.nn.softmax(attn_logits)

    attn = jnp.einsum('...htT,...Thd->...thd', attn_weights, v)

    # Concatenate attention matrix of all heads into a single vector.
    # Shape [..., q_index_dim, num_heads * attn_dim]
    attn = jnp.reshape(attn, (*single_electron_stream.shape[:-1], -1))

    # Apply a final projection to get the final embeddings.
    # Output shape: [..., q_index_dim, output_channels]
    return network_blocks.linear_layer(attn, params['attn_output'])

  return init, apply

def make_mlp(num_perceptron_per_layer: int = 1, activation_fct_name: str = "TANH") ->...:
  """Construct MLP, with final linear projection to embedding size."""
  if activation_fct_name.upper() == "TANH":
    activation_fct = jnp.tanh
  elif activation_fct_name.upper() == "ELU":
    activation_fct = jax.nn.elu
  elif activation_fct_name.upper() == "GELU":
    activation_fct = jax.nn.gelu

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
      return activation_fct(jnp.dot(inputs, params[0]['w']) + params[0]['b'])

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
        x = activation_fct(jnp.dot(inputs, params[i]['w']) + params[i]['b'])
      return x

    return init, apply

def make_self_attention_block(num_layers: int,
                              num_heads: int,
                              attn_dim: int,
                              value_dim: int,
                              mlp_dim: int,
                              num_perceptrons_per_layer: int,
                              use_layer_norm: bool = False,
                              mlp_activation_fct: str = "TANH") ->...:
  """Create a QKV self-attention block."""
  attention_init, attention_apply = make_multi_head_attention(
      num_heads, attn_dim, value_dim, mlp_dim)

  if use_layer_norm:
    layer_norm_init, layer_norm_apply = make_layer_norm()
  
  mlp_init, mlp_apply = make_mlp(num_perceptrons_per_layer, mlp_activation_fct)

  def init(key: chex.PRNGKey, mlp_dim: int) -> networks.ParamTree:
    params = {}
    attn_params = []
    ln_params = []
    mlp_params = []

    for _ in range(num_layers):
      key, attn_key, mlp_key = jax.random.split(key, 3)
      attn_params.append(
          attention_init(
              attn_key))
      if use_layer_norm:
        ln_params.append([layer_norm_init(mlp_dim), layer_norm_init(mlp_dim)])
      mlp_params.append(mlp_init(mlp_key, mlp_dim, mlp_dim))

    params['attention'] = attn_params
    params['ln'] = ln_params
    params['mlp'] = mlp_params

    return params

  def apply(params: networks.ParamTree, single_electron_stream: jnp.ndarray) -> jnp.ndarray:
    x = single_electron_stream
    for layer in range(num_layers):
      attn_output = attention_apply(params['attention'][layer], x)
      # Residual + optional LayerNorm.
      x = x + attn_output
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][0], x)

      # MLP
      mlp_output = mlp_apply(params['mlp'][layer], x)

      # Residual + optional LayerNorm.
      x = x + mlp_output
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][1], x)

    return x

  return init, apply

def make_psiformer_layers(
    nspins: Tuple[int, ...],
    options: PsiformerOptions,
) -> Tuple[networks.InitLayersFn, networks.ApplyLayersFn]:
  """Creates the permutation-equivariant layers for the Psiformer.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    options: network options.

  Returns:
    Tuple of init, apply functions.
  """
  del nspins  # Unused.

  # print(options)

  # Attention network.
  # input_attn_dim = options.num_heads * options.attn_dim
  self_attn_init, self_attn_apply = make_self_attention_block(
      num_layers=options.num_layers,
      num_heads=options.num_heads,
      attn_dim=options.attn_dim,
      value_dim=options.value_dim,
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

    # Map to Attention dim.
    key, subkey = jax.random.split(key)
    params['embed'] = network_blocks.init_linear_layer(
        subkey, in_dim=feature_dim, out_dim=options.mlp_dim, include_bias=False
    )['w']

    # Attention block params.
    key, subkey = jax.random.split(key)
    params.update(self_attn_init(key, options.mlp_dim))

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
    """Applies the Psiformer interaction layers to a walker configuration.

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
    # Only one-electron features are used by the Psiformer.
    ae_features, _ = options.feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input']
    )

    # For the Psiformer, the spin feature is required for correct permutation
    # equivariance.
    ae_features = jnp.concatenate((ae_features, spins[..., None]), axis=-1)

    features = ae_features  # Just 1-electron stream for now.

    # jax.debug.print("apply from make_psiformer_layers: [ae, feature] = {}", [ae, features])

    # Embed into attention dimension.
    x = jnp.dot(features, params['embed'])

    return self_attn_apply(params, x)

  return init, apply

def make_fermi_net(
    nspins: Tuple[int, ...],
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    complex_output: bool = False,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    # Psiformer-specific kwargs below.
    num_layers: int,
    num_heads: int,
    attn_dim: int,
    value_dim: int,
    mlp_dim: int,
    num_perceptrons_per_layer: int,
    use_layer_norm: bool,
    mlp_activation_fct: str,
    pbc_lattice: jnp.ndarray,
    use_spin_channels: bool = True,
) -> networks.Network:
  """Psiformer with stacked Self Attention layers.

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
    num_layers: Number of stacked self-attention layers.
    num_heads: Number of self-attention heads.
    attn_dim: Embedding dimension per-head for self-attention.
    mlp_dim: Tuple of hidden dimensions of the MLP.
    use_layer_norm: If true, use layer_norm on both attention and MLP.

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

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      envelope=envelope,
      feature_layer=feature_layer,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=True,  # Required for Psiformer.
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      attn_dim=attn_dim,
      value_dim=value_dim,
      mlp_dim=mlp_dim,
      num_perceptrons_per_layer=num_perceptrons_per_layer,
      use_layer_norm=use_layer_norm,
      mlp_activation_fct=mlp_activation_fct,
      pbc_lattice = pbc_lattice,
  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, options)

  orbitals_init, orbitals_apply = networks.make_orbitals(
      nspins=nspins,
      options=options,
      equivariant_layers=psiformer_layers,
      use_spin_channels=use_spin_channels,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Psiformer.

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
