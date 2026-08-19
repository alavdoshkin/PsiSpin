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
# limitations under the License.

"""Multiplicative envelope functions."""

import enum
from typing import Any, Mapping, Sequence, Union

import attr
import jax.numpy as jnp
from typing_extensions import Protocol


class EnvelopeType(enum.Enum):
  """The point at which the envelope is applied."""
  PRE_ORBITAL = enum.auto()
  PRE_DETERMINANT = enum.auto()


class EnvelopeLabel(enum.Enum):
  """Available multiplicative envelope functions."""
  NULL = enum.auto()


class EnvelopeInit(Protocol):

  def __call__(
      self, output_dims: Union[int, Sequence[int]], ndim: int
  ) -> Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]:
    """Returns the envelope parameters.

    Envelopes applied separately to each spin channel must create a sequence of
    parameters, one for each spin channel. Other envelope types must create a
    single mapping.

    Args:
      output_dims: The dimension of the layer to which the envelope is applied,
        per-spin channel for pre_determinant envelopes and a scalar otherwise.
      ndim: Dimension of system. Change with care.
    """


class EnvelopeApply(Protocol):

  def __call__(self, *, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
               **kwargs: jnp.ndarray) -> jnp.ndarray:
    """Returns a multiplicative envelope to ensure boundary conditions are met.

    If the envelope is applied before orbital shaping or after determinant
    evaluation, the envelope function is called once and N is the number of
    electrons. If the envelope is applied after orbital shaping and before
    determinant evaluation, the envelope function is called once per spin
    channel and N is the number of electrons in the spin channel.

    Args:
      ae: one-electron position features, shape (N, ndim).
      r_ae: one-electron position norms, shape (N, 1).
      r_ee: electron-electron distances, shape (N, nel, 1).
      **kwargs: learnable parameters of the envelope function.
    """


@attr.s(auto_attribs=True)
class Envelope:
  apply_type: EnvelopeType
  init: EnvelopeInit
  apply: EnvelopeApply


def make_null_envelope() -> Envelope:
  """Creates an no-op (identity) envelope."""

  def init(
      output_dims: Sequence[int], ndim: int = 3
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del ndim  # unused
    return [{} for _ in output_dims]

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray,
            r_ee: jnp.ndarray) -> jnp.ndarray:
    del ae, r_ae, r_ee
    return jnp.ones(shape=(1,))

  return Envelope(EnvelopeType.PRE_DETERMINANT, init, apply)


def get_envelope(
    envelope_label: EnvelopeLabel,
    **kwargs: Any,
) -> Envelope:
  """Gets the desired multiplicative envelope function.

  Args:
    envelope_label: envelope function required.
    **kwargs: keyword arguments forwarded to the envelope.

  Returns:
    (envelope_type, envelope), where envelope_type describes when the envelope
    should be applied in the network and envelope is the envelope function.
  """
  envelope_builders = {
      EnvelopeLabel.NULL: make_null_envelope,
  }
  return envelope_builders[envelope_label](**kwargs)
