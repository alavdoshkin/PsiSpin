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

"""Implementation of Fermionic Neural Network in JAX."""
import enum
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import attr
import chex
from psispin import envelopes
from psispin import network_blocks
import jax
import jax.numpy as jnp
from typing_extensions import Protocol

FermiLayers = Tuple[Tuple[int, int], ...]
# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet
ParamTree = Union[
    jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']
]
# pytype: enable=not-supported-yet
# Parameters for a single part of the network are just a dict.
Param = MutableMapping[str, jnp.ndarray]


@chex.dataclass
class FermiNetData:
  """Data passed to network.

  Shapes given for an unbatched element (i.e. a single MCMC configuration).

  NOTE:
    the networks are written in batchless form. Typically one then maps
    (pmap+vmap) over every attribute of FermiNetData (nb this is required if
    using KFAC, as it assumes the FIM is estimated over a batch of data), but
    this is not strictly required. If some attributes are not mapped, then JAX
    simply broadcasts them to the mapped dimensions (i.e. those attributes are
    treated as identical for every MCMC configuration.

  Attributes:
    positions: walker positions, shape (nelectrons*ndim).
    spins: spins of each walker, shape (nelectrons).
  """

  # We need to be able to construct instances of this with leaf nodes as jax
  # arrays (for the actual data) and as integers (to use with in_axes for
  # jax.vmap etc). We can type the struct to be either all arrays or all ints
  # using Generic, it just slightly complicates the type annotations in a few
  # functions (i.e. requires FermiNetData[jnp.ndarray] annotation).
  positions: Any
  spins: Any


## Interfaces (public) ##


class InitFermiNet(Protocol):

  def __call__(self, key: chex.PRNGKey) -> ParamTree:
    """Returns initialized parameters for the network.

    Args:
      key: RNG state
    """


class FermiNetLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      electrons: jnp.ndarray,
      spins: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the sign and log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
      spins: 1D array specifying the spin state of each electron.
    """


class LogFermiNetLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      electrons: jnp.ndarray,
      spins: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns the log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
      spins: 1D array specifying the spin state of each electron.
    """


class OrbitalFnLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.

    Returns:
      Sequence of orbitals.
    """


class InitLayersFn(Protocol):

  def __call__(self, key: chex.PRNGKey) -> Tuple[int, ParamTree]:
    """Returns output dim and initialized parameters for the interaction layers.

    Args:
      key: RNG state
    """


class ApplyLayersFn(Protocol):

  def __call__(
      self,
      params: ParamTree,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      spins: jnp.ndarray,
  ) -> jnp.ndarray:
    """Forward evaluation of the equivariant interaction layers.

    Args:
      params: parameters for the interaction and permutation-equivariant layers.
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


## Interfaces (network components) ##


class FeatureInit(Protocol):

  def __call__(self) -> Tuple[Tuple[int, int], Param]:
    """Creates the learnable parameters for the feature input layer.

    Returns:
      Tuple of ((x, y), params), where x and y are the number of one-electron
      features per electron and number of two-electron features per pair of
      electrons respectively, and params is a (potentially empty) mapping of
      learnable parameters associated with the feature construction layer.
    """


class FeatureApply(Protocol):

  def __call__(
      self,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      **params: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Creates the features to pass into the network.

    Args:
      ae: one-electron position features. Shape: (nelectron, ndim).
      r_ae: one-electron position norms. Shape: (nelectron, 1).
      ee: electron-electron vectors. Shape: (nelectron, nelectron, 3).
      r_ee: electron-electron distances. Shape: (nelectron, nelectron).
      **params: learnable parameters, as initialised in the corresponding
        FeatureInit function.
    """


@attr.s(auto_attribs=True)
class FeatureLayer:
  init: FeatureInit
  apply: FeatureApply


class FeatureLayerType(enum.Enum):
  STANDARD = enum.auto()


class MakeFeatureLayer(Protocol):

  def __call__(
      self,
      nspins: Sequence[int],
      ndim: int,
      **kwargs: Any,
  ) -> FeatureLayer:
    """Builds the FeatureLayer object.

    Args:
      nspins: tuple of the number of spin-up and spin-down electrons.
      ndim: dimension of the system.
      **kwargs: additional kwargs to use for creating the specific FeatureLayer.
    """


## Network settings ##


@attr.s(auto_attribs=True, kw_only=True)
class BaseNetworkOptions:
  """Options controlling the overall network architecture.

  Attributes:
    ndim: dimension of system. Change only with caution.
    determinants: Number of determinants to use.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    envelope: Envelope object to create and apply the multiplicative envelope.
    feature_layer: Feature object to create and apply the input features for the
      one- and two-electron layers.
    complex_output: If true, the network outputs complex numbers.
  """

  ndim: int = 3
  determinants: int = 16
  states: int = 0
  full_det: bool = True
  rescale_inputs: bool = False
  bias_orbitals: bool = False
  envelope: envelopes.Envelope = attr.ib(
      default=attr.Factory(
          envelopes.make_null_envelope,
          takes_self=False))
  feature_layer: FeatureLayer = None
  complex_output: bool = False
  pbc_lattice: jnp.ndarray = jnp.array([])


@attr.s(auto_attribs=True, kw_only=True)
class FermiNetOptions(BaseNetworkOptions):
  """Options controlling the FermiNet architecture.

  Attributes:
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    separate_spin_channels: If True, use separate two-electron streams for
      spin-parallel and spin-antiparallel  pairs of electrons. If False, use the
      same stream for all pairs of electrons.
    schnet_electron_electron_convolutions: Tuple of embedding dimension to use
      for a SchNet-style convolution between the one- and two-electron streams
      at each layer of the network. If empty, the original FermiNet embedding is
      used.
    use_last_layer: If true, the outputs of the one- and two-electron streams
      are combined into permutation-equivariant features and passed into the
      final orbital-shaping layer. Otherwise, just the output of the
      one-electron stream is passed into the orbital-shaping layer.
  """

  hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32), (256, 32))
  separate_spin_channels: bool = False
  schnet_electron_electron_convolutions: Tuple[int, ...] = ()
  use_last_layer: bool = False


# Network class.


@attr.s(auto_attribs=True)
class Network:
  options: BaseNetworkOptions
  init: InitFermiNet
  apply: FermiNetLike
  orbitals: OrbitalFnLike


# Internal utilities


def _split_spin_pairs(
    arr: jnp.ndarray,
    nspins: Tuple[int, int],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Splits array into parallel and anti-parallel spin channels.

  For an array of dimensions (nelec, nelec, ...), where nelec = sum(nspins),
  and the first nspins[0] elements along the first two axes correspond to the up
  electrons, we have an array like:

    up,up   | up,down
    down,up | down,down

  Split this into the diagonal and off-diagonal blocks. As nspins[0] !=
  nspins[1] in general, flatten the leading two dimensions before combining the
  blocks.

  Args:
    arr: array with leading dimensions (nelec, nelec).
    nspins: number of electrons in each spin channel.

  Returns:
    parallel, antiparallel arrays, where
       - parallel is of shape (nspins[0]**2 + nspins[1]**2, ...) and the first
         nspins[0]**2 elements correspond to the up,up block and the subsequent
         elements to the down,down block.
       - antiparallel is of shape (2 * nspins[0] + nspins[1], ...) and the first
         nspins[0] + nspins[1] elements correspond to the up,down block and the
         subsequent
         elements to the down,up block.
  """
  if len(nspins) != 2:
    raise ValueError(
        'Separate spin channels has not been verified with spin sampling.'
    )
  up_up, up_down, down_up, down_down = network_blocks.split_into_blocks(
      arr, nspins
  )
  trailing_dims = jnp.shape(arr)[2:]
  parallel_spins = [
      up_up.reshape((-1,) + trailing_dims),
      down_down.reshape((-1,) + trailing_dims),
  ]
  antiparallel_spins = [
      up_down.reshape((-1,) + trailing_dims),
      down_up.reshape((-1,) + trailing_dims),
  ]
  return (
      jnp.concatenate(parallel_spins, axis=0),
      jnp.concatenate(antiparallel_spins, axis=0),
  )


def _combine_spin_pairs(
    parallel_spins: jnp.ndarray,
    antiparallel_spins: jnp.ndarray,
    nspins: Tuple[int, int],
) -> jnp.ndarray:
  """Combines arrays of parallel spins and antiparallel spins.

  This is the reverse of _split_spin_pairs.

  Args:
    parallel_spins: array of shape (nspins[0]**2 + nspins[1]**2, ...).
    antiparallel_spins: array of shape (2 * nspins[0] * nspins[1], ...).
    nspins: number of electrons in each spin channel.

  Returns:
    array of shape (nelec, nelec, ...).
  """
  if len(nspins) != 2:
    raise ValueError(
        'Separate spin channels has not been verified with spin sampling.'
    )
  nsame_pairs = [nspin**2 for nspin in nspins]
  same_pair_partitions = network_blocks.array_partitions(nsame_pairs)
  up_up, down_down = jnp.split(parallel_spins, same_pair_partitions, axis=0)
  up_down, down_up = jnp.split(antiparallel_spins, 2, axis=0)
  trailing_dims = jnp.shape(parallel_spins)[1:]
  up = jnp.concatenate(
      (
          up_up.reshape((nspins[0], nspins[0]) + trailing_dims),
          up_down.reshape((nspins[0], nspins[1]) + trailing_dims),
      ),
      axis=1,
  )
  down = jnp.concatenate(
      (
          down_up.reshape((nspins[1], nspins[0]) + trailing_dims),
          down_down.reshape((nspins[1], nspins[1]) + trailing_dims),
      ),
      axis=1,
  )
  return jnp.concatenate((up, down), axis=0)


## Network layers: features ##
def make_periodic_construct_input_features(pbc_lattice: jnp.ndarray) -> ...:
  # WARNING: Using periodic norm for construct_input_features distorts the distances r_ee, r_ae on
  # length scales of the unit cell boundary. 
  # It only agrees with the real-space norm on short length scales.
  print("making construct_input_features with periodic norm")
  lattice = pbc_lattice
  rec = 2 * jnp.pi * jnp.linalg.inv(lattice)
  lattice_metric =  lattice.T @ lattice

  # periodic_norm copied from psispin.pbc.feature_layer
  def periodic_norm(metric: jnp.ndarray, scaled_r: jnp.ndarray) -> jnp.ndarray:
    """Returns the periodic norm of a set of vectors.

    Args:
      metric: metric tensor in fractional coordinate system, A.T A, where A is the
        lattice vectors.
      scaled_r: vectors in fractional coordinates of the lattice cell, with
        trailing dimension ndim, to compute the periodic norm of.
    """
    chex.assert_rank(metric, expected_ranks=2)
    a = (1 - jnp.cos(2 * jnp.pi * scaled_r))
    b = jnp.sin(2 * jnp.pi * scaled_r)
    cos_term = jnp.einsum('...m,mn,...n->...', a, metric, a)
    sin_term = jnp.einsum('...m,mn,...n->...', b, metric, b)
    return (1 / (2 * jnp.pi)) * jnp.sqrt(cos_term + sin_term)

  def construct_input_features(
      pos: jnp.ndarray,
      ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Constructs inputs to Fermi Net from raw electron positions.

    Args:
      pos: electron positions. Shape (nelectrons*ndim,).
      ndim: dimension of system. Change only with caution.

    Returns:
      ae, ee, r_ae, r_ee tuple, where:
        ae: one-electron position features. Shape (nelectron, ndim).
        ee: electron-electron vector. Shape (nelectron, nelectron, ndim).
        r_ae: one-electron position norm. Shape (nelectron, 1).
        r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
      The diagonal terms in r_ee are masked out such that the gradients of these
      terms are also zero.
    """
    ae = jnp.reshape(pos, [-1, ndim])
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    # convert to position in fraction of unit cell lattice vectors for application of periodic norm
    s_ae = jnp.einsum('il,jl->ji', rec / 2 / jnp.pi, ae)
    s_ee = jnp.einsum('il,jkl->jki', rec / 2 / jnp.pi, ee)

    r_ae = periodic_norm(lattice_metric, s_ae)[..., None]
    # Avoid computing the norm of zero, as it has undefined grad
    n = ee.shape[0]
    r_ee = (
        periodic_norm(lattice_metric, s_ee + jnp.eye(n)[..., None]) * (1.0 - jnp.eye(n)))

    return ae, ee, r_ae, r_ee[..., None]

  return construct_input_features

# Default construct input features used with default Hamiltonian
def construct_input_features(
    pos: jnp.ndarray,
    ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Constructs inputs to Fermi Net from raw electron positions.

  Args:
    pos: electron positions. Shape (nelectrons*ndim,).
    ndim: dimension of system. Change only with caution.

  Returns:
    ae, ee, r_ae, r_ee tuple, where:
      ae: one-electron position features. Shape (nelectron, ndim).
      ee: electron-electron vector. Shape (nelectron, nelectron, ndim).
      r_ae: one-electron position norm. Shape (nelectron, 1).
      r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
    The diagonal terms in r_ee are masked out such that the gradients of these
    terms are also zero.
  """
  ae = jnp.reshape(pos, [-1, ndim])
  ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])

  r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
  # Avoid computing the norm of zero, as it has undefined grad
  n = ee.shape[0]
  r_ee = (
      jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
  return ae, ee, r_ae, r_ee[..., None]


def make_ferminet_features(
    nspins: Optional[Tuple[int, int]] = None,
    ndim: int = 3,
    rescale_inputs: bool = False,
) -> FeatureLayer:
  """Returns the init and apply functions for the standard features."""

  del nspins

  def init() -> Tuple[Tuple[int, int], Param]:
    return (ndim + 1, ndim + 1), {}

  def apply(ae, r_ae, ee, r_ee) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if rescale_inputs:
      log_r_ae = jnp.log(1 + r_ae)  # grows as log(r) rather than r
      ae_features = jnp.concatenate((log_r_ae, ae * log_r_ae / r_ae), axis=-1)

      log_r_ee = jnp.log(1 + r_ee)
      ee_features = jnp.concatenate((log_r_ee, ee * log_r_ee / r_ee), axis=2)

    else:
      ae_features = jnp.concatenate((r_ae, ae), axis=-1)
      ee_features = jnp.concatenate((r_ee, ee), axis=2)
    return ae_features, ee_features

  return FeatureLayer(init=init, apply=apply)


## Network layers: permutation-equivariance ##


def construct_symmetric_features(
    h_one: jnp.ndarray,
    h_two: jnp.ndarray,
    nspins: Tuple[int, int],
    h_aux: Optional[jnp.ndarray],
) -> jnp.ndarray:
  """Combines intermediate features from rank-one and -two streams.

  Args:
    h_one: set of one-electron features. Shape: (nelectrons, n1), where n1 is
      the output size of the previous layer.
    h_two: set of two-electron features. Shape: (nelectrons, nelectrons, n2),
      where n2 is the output size of the previous layer.
    nspins: Number of spin-up and spin-down electrons.
    h_aux: optional auxiliary features to include. Shape (nelectrons, naux).

  Returns:
    array containing the permutation-equivariant features: the input set of
    one-electron features, the mean of the one-electron features over each
    (occupied) spin channel, and the mean of the two-electron features over each
    (occupied) spin channel. Output shape (nelectrons, 3*n1 + 2*n2 + naux) if
    there are both spin-up and spin-down electrons and
    (nelectrons, 2*n1 + n2 + naux) otherwise.
  """
  # Split features into spin up and spin down electrons
  spin_partitions = network_blocks.array_partitions(nspins)
  h_ones = jnp.split(h_one, spin_partitions, axis=0)
  h_twos = jnp.split(h_two, spin_partitions, axis=0)

  # Construct inputs to next layer
  # h.size == 0 corresponds to unoccupied spin channels.
  g_one = [jnp.mean(h, axis=0, keepdims=True) for h in h_ones if h.size > 0]
  g_one = [jnp.tile(g, [h_one.shape[0], 1]) for g in g_one]

  g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]

  features = [h_one] + g_one + g_two
  if h_aux is not None:
    features.append(h_aux)
  return jnp.concatenate(features, axis=1)


## Network layers: main layers ##


def make_schnet_convolution(
    nspins: Tuple[int, int], separate_spin_channels: bool
) -> ...:
  """Returns init/apply pair for SchNet-style convolutions.

  See Gerard et al, arXiv:2205.09438.

  Args:
    nspins: number of electrons in each spin channel.
    separate_spin_channels: If True, treat pairs of spin-parallel and
      spin-antiparallel electrons with separate  embeddings. If False, use the
      same embedding for all pairs.
  """

  def init(
      key: chex.PRNGKey, dims_one: int, dims_two: int, embedding_dim: int
  ) -> ParamTree:
    """Returns parameters for learned Schnet convolutions.

    Args:
      key: PRNG state.
      dims_one: number of hidden units of the one-electron layer.
      dims_two: number of hidden units of the two-electron layer.
      embedding_dim: embedding dimension to use for the convolution.
    """
    nchannels = 2 if separate_spin_channels else 1
    key_one, *key_two = jax.random.split(key, num=nchannels + 1)
    h_one_kernel = network_blocks.init_linear_layer(
        key_one, in_dim=dims_one, out_dim=embedding_dim, include_bias=False
    )
    h_two_kernels = []
    for i in range(nchannels):
      h_two_kernels.append(
          network_blocks.init_linear_layer(
              key_two[i],
              in_dim=dims_two,
              out_dim=embedding_dim,
              include_bias=False,
          )
      )
    return {
        'single': h_one_kernel['w'],
        'double': [kernel['w'] for kernel in h_two_kernels],
    }

  def apply(
      params: ParamTree, h_one: jnp.ndarray, h_two: Tuple[jnp.ndarray, ...]
  ) -> jnp.ndarray:
    """Applies the convolution B h_two . C h_one."""
    # Two distinctions from Gerard et al. They give the electron-electron
    # embedding in Eq 6 as
    # \sum_j B_{sigma_{ij}}(h_{ij} * C_{sigma_{ij}}(h_{j}
    # ie the C kernel is also dependent upon the spin pair. This does not match
    # the definition in the PauliNet paper. We keep the C kernel independent of
    # spin pair, and make B dependent upon spin-pair if separate_spin_channels
    # is True.
    # This (and Eq 5) gives that all j electrons are summed over, whereas
    # FermiNet concatenates the sum over spin up and spin-down electrons
    # separately. We follow the latter always.
    # These changes are in keeping with the spirit of FermiNet and SchNet
    # convolutions, if not the detail provided by Gerard et al.
    h_one_embedding = network_blocks.linear_layer(h_one, params['single'])
    h_two_embeddings = [
        network_blocks.linear_layer(h_two_channel, layer_param)
        for h_two_channel, layer_param in zip(h_two, params['double'])
    ]
    if separate_spin_channels:
      # h_two is a tuple of parallel spin pairs and anti-parallel spin pairs.
      h_two_embedding = _combine_spin_pairs(
          h_two_embeddings[0], h_two_embeddings[1], nspins
      )
    else:
      h_two_embedding = h_two_embeddings[0]
    return h_one_embedding * h_two_embedding

  return init, apply


def make_fermi_net_layers(
    nspins: Tuple[int, int], options: FermiNetOptions
) -> Tuple[InitLayersFn, ApplyLayersFn]:
  """Creates the permutation-equivariant and interaction layers for FermiNet.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    options: network options.

  Returns:
    Tuple of init, apply functions.
  """

  schnet_electron_init, schnet_electron_apply = make_schnet_convolution(
      nspins=nspins, separate_spin_channels=options.separate_spin_channels
  )

  if all(
      len(hidden_dims) != len(options.hidden_dims[0])
      for hidden_dims in options.hidden_dims
  ):
    raise ValueError(
        'Each layer does not have the same number of streams: '
        f'{options.hidden_dims}'
    )

  if options.use_last_layer:
    num_convolutions = len(options.hidden_dims) + 1
  else:
    num_convolutions = len(options.hidden_dims)
  if (
      options.schnet_electron_electron_convolutions
      and len(options.schnet_electron_electron_convolutions) != num_convolutions
  ):
    raise ValueError(
        'Inconsistent number of layers for convolution and '
        'one- and two-electron streams. '
        f'{len(options.schnet_electron_electron_convolutions)=}, '
        f'expected {num_convolutions} layers.'
    )
  def init(key: chex.PRNGKey) -> Tuple[int, ParamTree]:
    """Returns tuple of output dimension from the final layer and parameters."""

    params = {}
    (num_one_features, num_two_features), params['input'] = (
        options.feature_layer.init()
    )

    # The input to layer L of the one-electron stream is from
    # construct_symmetric_features and shape (nelectrons, nfeatures), where
    # nfeatures is
    # i) output from the previous one-electron layer (out1);
    # ii) the mean for each spin channel from each layer (out1 * # channels);
    # iii) the mean for each spin channel from each two-electron layer (out2 * #
    # channels)
    # iv) any additional features from auxiliary streams.
    # We don't create features for spin channels
    # which contain no electrons (i.e. spin-polarised systems).
    nchannels = len([nspin for nspin in nspins if nspin > 0])

    def nfeatures(out1, out2, aux):
      return (nchannels + 1) * out1 + nchannels * out2 + aux

    # one-electron stream, per electron:
    #  - one-electron features per atom (default: electron-atom vectors
    #    (ndim/atom) and distances (1/atom)),
    # two-electron stream, per pair of electrons:
    #  - two-electron features per electron pair (default: electron-electron
    #    vector (dim) and distance (1))
    dims_one_in = num_one_features
    dims_two_in = num_two_features
    # Note SchNet-style convolution with a electron-nuclear stream assumes
    # FermiNet features currently.

    key, subkey = jax.random.split(key)
    layers = []
    for i in range(len(options.hidden_dims)):
      layer_params = {}
      key, single_key, *double_keys = jax.random.split(key, num=4)

      # Learned convolution on each layer.
      if options.schnet_electron_electron_convolutions:
        key, subkey = jax.random.split(key)
        layer_params['schnet'] = schnet_electron_init(
            subkey,
            dims_one=dims_one_in,
            dims_two=dims_two_in,
            embedding_dim=options.schnet_electron_electron_convolutions[i],
        )
        dims_two_embedding = options.schnet_electron_electron_convolutions[i]
      else:
        dims_two_embedding = dims_two_in

      dims_one_in = nfeatures(dims_one_in, dims_two_embedding, 0)

      # Layer initialisation
      dims_one_out, dims_two_out = options.hidden_dims[i]
      layer_params['single'] = network_blocks.init_linear_layer(
          single_key,
          in_dim=dims_one_in,
          out_dim=dims_one_out,
          include_bias=True,
      )

      if i < len(options.hidden_dims) - 1 or options.use_last_layer:
        ndouble_channels = 2 if options.separate_spin_channels else 1
        layer_params['double'] = []
        for ichannel in range(ndouble_channels):
          layer_params['double'].append(
              network_blocks.init_linear_layer(
                  double_keys[ichannel],
                  in_dim=dims_two_in,
                  out_dim=dims_two_out,
                  include_bias=True,
              )
          )
        if not options.separate_spin_channels:
          # Just have a single dict rather than a list of length 1 to match
          # older behaviour (when one stream was used for all electron pairs).
          layer_params['double'] = layer_params['double'][0]

      layers.append(layer_params)
      dims_one_in = dims_one_out
      dims_two_in = dims_two_out

    if options.use_last_layer:
      layers.append({})
      # Pass symmetric features to the orbital shaping layer.
      if options.schnet_electron_electron_convolutions:
        key, subkey = jax.random.split(key)
        layers[-1]['schnet'] = schnet_electron_init(
            subkey,
            dims_one=dims_one_in,
            dims_two=dims_two_in,
            embedding_dim=options.schnet_electron_electron_convolutions[-1],
        )
        dims_two_in = options.schnet_electron_electron_convolutions[-1]
      output_dims = nfeatures(dims_one_in, dims_two_in, 0)
    else:
      # Pass output of the one-electron stream straight to orbital shaping.
      output_dims = dims_one_in

    params['streams'] = layers

    return output_dims, params

  def electron_electron_convolution(
      params: ParamTree,
      h_one: jnp.ndarray,
      h_two: Tuple[jnp.ndarray, ...],
  ) -> jnp.ndarray:
    if options.schnet_electron_electron_convolutions:
      # SchNet-style embedding: convolve embeddings of one- and two-electron
      # streams.
      h_two_embedding = schnet_electron_apply(params['schnet'], h_one, h_two)
    elif options.separate_spin_channels:
      # FermiNet embedding from separate spin channels for parallel and
      # anti-parallel pairs of spins. Need to reshape and combine spin channels.
      h_two_embedding = _combine_spin_pairs(h_two[0], h_two[1], nspins)
    else:
      # Original FermiNet embedding.
      h_two_embedding = h_two[0]
    return h_two_embedding

  def apply_layer(
      params: Mapping[str, ParamTree],
      h_one: jnp.ndarray,
      h_two: Tuple[jnp.ndarray, ...],
  ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
    if options.separate_spin_channels:
      assert len(h_two) == 2
    else:
      assert len(h_two) == 1

    residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y

    # Permutation-equivariant block.
    h_two_embedding = electron_electron_convolution(params, h_one, h_two)
    h_one_in = construct_symmetric_features(
        h_one, h_two_embedding, nspins, h_aux=None
    )

    # Execute next layer.
    h_one_next = jnp.tanh(
        network_blocks.linear_layer(h_one_in, **params['single'])
    )
    h_one = residual(h_one, h_one_next)
    # Only perform the auxiliary streams if parameters are present (ie not the
    # final layer of the network if use_last_layer is False).
    if 'double' in params:
      if options.separate_spin_channels:
        params_double = params['double']
      else:
        # Using one stream for pairs of electrons. Make a sequence of params of
        # same length as h_two.
        params_double = [params['double']]
      h_two_next = [
          jnp.tanh(network_blocks.linear_layer(prev, **param))
          for prev, param in zip(h_two, params_double)
      ]
      h_two = tuple(residual(prev, new) for prev, new in zip(h_two, h_two_next))

    return h_one, h_two

  def apply(
      params,
      *,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      spins: jnp.ndarray,
  ) -> jnp.ndarray:
    """Applies the FermiNet interaction layers to a walker configuration.

    Args:
      params: parameters for the interaction and permutation-equivariant layers.
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
    del spins  # Unused.

    ae_features, ee_features = options.feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input']
    )

    h_one = ae_features  # single-electron features

    if options.separate_spin_channels:
      # Use the same stream for spin-parallel and spin-antiparallel electrons.
      # In order to handle different numbers of spin-up and spin-down electrons,
      # flatten the i,j indices.
      # Shapes: (nup*nup + ndown*ndown, nfeatures), (nup*down*2, nfeatures).
      h_two = _split_spin_pairs(ee_features, nspins)
    else:
      # Use the same stream for spin-parallel and spin-antiparallel electrons.
      # Keep as 3D array to make splitting over spin channels in
      # construct_symmetric_features simple.
      # Shape: (nelectron, nelectron, nfeatures)
      h_two = [ee_features]

    for i in range(len(options.hidden_dims)):
      h_one, h_two = apply_layer(
          params['streams'][i],
          h_one,
          h_two,
      )

    if options.use_last_layer:
      last_layer = params['streams'][-1]
      h_two_embedding = electron_electron_convolution(last_layer, h_one, h_two)
      h_to_orbitals = construct_symmetric_features(
          h_one, h_two_embedding, nspins, h_aux=None
      )
    else:
      # Didn't apply the final two-electron and auxiliary layers. Just forward
      # the output of the one-electron stream to the orbital projection layer.
      h_to_orbitals = h_one

    return h_to_orbitals

  return init, apply


## Network layers: orbitals ##


def make_orbitals(
    nspins: Tuple[int, int],
    options: BaseNetworkOptions,
    equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn],
    use_spin_channels: bool = True,
) -> ...:
  """Returns init, apply pair for orbitals.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    options: Network configuration.
    equivariant_layers: Tuple of init, apply functions for the equivariant
      interaction part of the network.
  """

  equivariant_layers_init, equivariant_layers_apply = equivariant_layers

  # construct periodic input features so that r_ee is taken with periodic norm
  construct_input_features = make_periodic_construct_input_features(options.pbc_lattice)

  if not use_spin_channels:  # gets rid of spin channels
      nspins = (sum(nspins), 0)

  def init(key: chex.PRNGKey) -> ParamTree:
    """Returns initial random parameters for creating orbitals.

    Args:
      key: RNG state.
    """
    key, subkey = jax.random.split(key)
    params = {}
    dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)
    if nchannels == 0:
      raise ValueError('No electrons present!')

    # How many spin-orbitals do we need to create per spin channel?
    nspin_orbitals = []
    num_states = max(options.states, 1)
    for nspin in active_spin_channels:
      if options.full_det:
        # Dense determinant. Need N orbitals per electron per determinant.
        norbitals = sum(nspins) * options.determinants * num_states
      else:
        # Spin-factored block-diagonal determinant. Need nspin orbitals per
        # electron per determinant.
        norbitals = nspin * options.determinants * num_states
      if options.complex_output:
        norbitals *= 2  # one output is real, one is imaginary
      nspin_orbitals.append(norbitals)

    # create envelope params
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      # Applied to output from final layer of 1e stream.
      output_dims = dims_orbital_in
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      # Applied to orbitals.
      if options.complex_output:
        output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
      else:
        output_dims = nspin_orbitals
    else:
      raise ValueError('Unknown envelope type')
    params['envelope'] = options.envelope.init(
        output_dims=output_dims, ndim=options.ndim
    )

    # orbital shaping
    orbitals = []
    for nspin_orbital in nspin_orbitals:
      key, subkey = jax.random.split(key)
      orbitals.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_orbital_in,
              out_dim=nspin_orbital,
              include_bias=options.bias_orbitals,
          )
      )
    params['orbital'] = orbitals

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.

    Returns:
      One matrix (two matrices if options.full_det is False) that exchange
      columns under the exchange of inputs of shape (ndet, nalpha+nbeta,
      nalpha+nbeta) (or (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)).
    """
    ae, ee, r_ae, r_ee = construct_input_features(pos, ndim=options.ndim)
    h_to_orbitals = equivariant_layers_apply(
        params['layers'],
        ae=ae,
        r_ae=r_ae,
        ee=ee,
        r_ee=r_ee,
        spins=spins,
    )

    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      envelope_factor = options.envelope.apply(
          ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
      )
      h_to_orbitals = envelope_factor * h_to_orbitals
    # Note split creates arrays of size 0 for spin channels without electrons.
    h_to_orbitals = jnp.split(
        h_to_orbitals, network_blocks.array_partitions(nspins), axis=0
    )
    # Drop unoccupied spin channels
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    active_spin_channels = [spin for spin in nspins if spin > 0]
    active_spin_partitions = network_blocks.array_partitions(
        active_spin_channels
    )
    # Create orbitals.
    orbitals = [
        network_blocks.linear_layer(h, **p)
        for h, p in zip(h_to_orbitals, params['orbital'])
    ]
    if options.complex_output:
      # create imaginary orbitals
      orbitals = [
          orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals
      ]

    # Apply envelopes if required.
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
      r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
      r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
      for i in range(len(active_spin_channels)):
        orbitals[i] = orbitals[i] * options.envelope.apply(
            ae=ae_channels[i],
            r_ae=r_ae_channels[i],
            r_ee=r_ee_channels[i],
            **params['envelope'][i],
        )

    # Reshape into matrices.
    shapes = [
        (spin, -1, sum(nspins) if options.full_det else spin)
        for spin in active_spin_channels
    ]
    orbitals = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
    ]
    orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    if options.full_det:
      orbitals = [jnp.concatenate(orbitals, axis=1)]

    return orbitals

  return init, apply


## Excited States  ##


def make_state_matrix(signed_network: FermiNetLike, n: int) -> FermiNetLike:
  """Construct a matrix-output ansatz which gives the Slater matrix of states.

  Let signed_network(params, pos, spins, options) be a function which returns
  psi_1(pos), psi_2(pos), ... psi_n(pos) as a pair of arrays of length n, one
  with values of sign(psi_k), one with values of log(psi_k). Then this function
  returns a new function which computes the matrix psi_i(pos_j), given an array
  of positions (and possibly spins) which has n times as many dimensions as
  expected by signed_network. The output of this new meta-matrix is also given
  as a sign, log pair.

  Args:
    signed_network: A function with the same calling convention as the FermiNet.
    n: the number of excited states, needed to know how to shape the determinant

  Returns:
    A function with two outputs which combines the individual excited states
    into a matrix of wavefunctions, one with the sign and one with the log.
  """

  def state_matrix(params, pos, spins):
    """Evaluate state_matrix for a given ansatz."""
    # `pos` has shape (n*nelectron*ndim), but can be reshaped as
    # (n, nelectron, ndim), that is, the first dimension indexes which excited
    # state we are considering, the second indexes electrons, and the third
    # indexes spatial dimensions. `spins` has the same ordering of indices,
    # but does not have the spatial dimensions.
    pos_ = jnp.reshape(pos, [n, -1])
    spins_ = jnp.reshape(spins, [n, -1])
    vmap_network = jax.vmap(signed_network, (None, 0, 0))
    sign_mat, log_mat = vmap_network(params, pos_, spins_)
    return sign_mat, log_mat

  return state_matrix


def make_state_trace(signed_network: FermiNetLike, n: int) -> FermiNetLike:
  """Construct a single-output f'n which gives the trace over the state matrix.

  Returns the sum of the diagonal of the matrix of log|psi| values created by
  make_state_matrix. That means for a set of inputs x_1, ..., x_n, instead of
  returning the full matrix of psi_i(x_j), only return the sum of the diagonal
  sum_i log(psi_i(x_i)), so one state per input. Used for MCMC sampling.

  Args:
    signed_network: A function with the same calling convention as the FermiNet.
    n: the number of excited states, needed to know how to shape the determinant

  Returns:
    A function with a multiple outputs which takes a set of inputs and returns
    one output per input.
  """
  state_matrix = make_state_matrix(signed_network, n)

  def state_trace(params, pos, spins, **kwargs):
    """Evaluate trace of the state matrix for a given ansatz."""
    _, log_in = state_matrix(params, pos, spins, **kwargs)

    return jnp.trace(log_in)

  return state_trace


def make_total_ansatz(signed_network: FermiNetLike,
                      n: int,
                      complex_output: bool = False) -> FermiNetLike:
  """Construct a single-output ansatz which gives the meta-Slater determinant.

  Let signed_network(params, pos, spins, options) be a function which returns
  psi_1(pos), psi_2(pos), ... psi_n(pos) as a pair of arrays, one with values
  of sign(psi_k), one with values of log(psi_k). Then this function returns a
  new function which computes det[psi_i(pos_j)], given an array of positions
  (and possibly spins) which has n times as many dimensions as expected by
  signed_network. The output of this new meta-determinant is also given as a
  sign, log pair.

  Args:
    signed_network: A function with the same calling convention as the FermiNet.
    n: the number of excited states, needed to know how to shape the determinant
    complex_output: If true, the output of the network is complex, and the
      individual states return phase angles rather than signs.

  Returns:
    A function with a single output which combines the individual excited states
    into a greater wavefunction given by the meta-Slater determinant.
  """
  state_matrix = make_state_matrix(signed_network, n)

  def total_ansatz(params, pos, spins, **kwargs):
    """Evaluate meta_determinant for a given ansatz."""
    sign_in, log_in = state_matrix(params, pos, spins, **kwargs)

    logmax = jnp.max(log_in)  # logsumexp trick
    if complex_output:
      # sign_in is a phase angle rather than a sign for complex networks
      mat_in = jnp.exp(log_in + 1.j * sign_in - logmax)
      sign_out, log_out = jnp.linalg.slogdet(mat_in)
      sign_out = jnp.angle(sign_out)
    else:
      sign_out, log_out = jnp.linalg.slogdet(sign_in * jnp.exp(log_in - logmax))
    log_out += n * logmax
    return sign_out, log_out

  return total_ansatz


## FermiNet ##


def make_fermi_net(
    nspins: Tuple[int, int],
    *,
    ndim: int = 3,
    determinants: int = 16,
    states: int = 0,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[FeatureLayer] = None,
    complex_output: bool = False,
    bias_orbitals: bool = False,
    full_det: bool = True,
    rescale_inputs: bool = False,
    # FermiNet-specific kwargs below.
    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    use_last_layer: bool = False,
    separate_spin_channels: bool = False,
    schnet_electron_electron_convolutions: Tuple[int, ...] = tuple(),
    pbc_lattice: jnp.ndarray,
) -> Network:
  """Creates functions for initializing parameters and evaluating psispin.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    ndim: dimension of system. Change only with caution.
    determinants: Number of determinants to use.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    complex_output: If true, the network outputs complex numbers.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    use_last_layer: If true, the outputs of the one- and two-electron streams
      are combined into permutation-equivariant features and passed into the
      final orbital-shaping layer. Otherwise, just the output of the
      one-electron stream is passed into the orbital-shaping layer.
    separate_spin_channels: Use separate learnable parameters for pairs of
      spin-parallel and spin-antiparallel electrons.
    schnet_electron_electron_convolutions: Dimension of embeddings used for
      electron-electron SchNet-style convolutions.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network. If
    options.states > 1, the length of the vectors returned by apply are equal
    to the number of states.
  """
  if sum([nspin for nspin in nspins if nspin > 0]) == 0:
    raise ValueError('No electrons present!')

  if not envelope:
    envelope = envelopes.make_null_envelope()

  if not feature_layer:
    feature_layer = make_ferminet_features(
        nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  options = FermiNetOptions(
      ndim=ndim,
      determinants=determinants,
      states=states,
      rescale_inputs=rescale_inputs,
      envelope=envelope,
      feature_layer=feature_layer,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      full_det=full_det,
      hidden_dims=hidden_dims,
      separate_spin_channels=separate_spin_channels,
      schnet_electron_electron_convolutions=schnet_electron_electron_convolutions,
      use_last_layer=use_last_layer,
      pbc_lattice = pbc_lattice,
  )

  if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    if options.bias_orbitals:
      raise ValueError('Cannot bias orbitals w/STO envelope.')

  equivariant_layers = make_fermi_net_layers(nspins, options)

  orbitals_init, orbitals_apply = make_orbitals(
      nspins=nspins,
      options=options,
      equivariant_layers=equivariant_layers,
  )

  def init(key: chex.PRNGKey) -> ParamTree:
    key, subkey = jax.random.split(key, num=2)
    return orbitals_init(subkey)

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network for a single datum.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute of the network evaluated at x.
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

  return Network(
      options=options, init=init, apply=apply, orbitals=orbitals_apply
  )
