# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training a CNN on MNIST with Keras and the DP SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

tf.config.threading.set_intra_op_parallelism_threads(3)

def random_choice_cond(x, size):
    tensor_size = tf.size(x)
    indices = tf.range(0, tensor_size, dtype=tf.int64)
    if size == 0:
        sample_flatten_index = tf.random.shuffle(indices)[:]
    else:     
        sample_flatten_index = tf.random.shuffle(indices)[:size]
    sample_index = tf.transpose(tf.unravel_index(tf.cast(sample_flatten_index,tf.int32), tf.shape(input=x))) #[Result: [indexes for the first sample], [indexes for the second sample]...]
    cond = tf.scatter_nd(sample_index, tf.ones(tf.shape(input=sample_index)[0],dtype=tf.bool), tf.shape(input=x))
    return cond

    # we need noise on the same fixed number of model parameters for each layer.

from tensorflow_privacy.privacy.dp_query import gaussian_query
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

def make_fixed_keras_optimizer_class(cls):
  """Constructs a DP Keras optimizer class from an existing one."""

  class FixedDPOptimizerClass(cls):
    """Differentially private subclass of given class cls.
    The class tf.keras.optimizers.Optimizer has two methods to compute
    gradients, `_compute_gradients` and `get_gradients`. The first works
    with eager execution, while the second runs in graph mode and is used
    by canned estimators.
    Internally, DPOptimizerClass stores hyperparameters both individually
    and encapsulated in a `GaussianSumQuery` object for these two use cases.
    However, this should be invisible to users of this class.
    
    
    btw. support negative num_parameters, used for all but num_parameters noises.
    """

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        num_parameters = 10,
        #noise_layer_type = "linear",
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initialize the DPOptimizerClass.
      Args:
        l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients)
        noise_multiplier: Ratio of the standard deviation to the clipping norm
        num_microbatches: The number of microbatches into which each minibatch
          is split.
      """
      super(FixedDPOptimizerClass, self).__init__(*args, **kwargs)
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier
      self._num_microbatches = num_microbatches
      self._dp_sum_query = gaussian_query.GaussianSumQuery(
          l2_norm_clip, l2_norm_clip * noise_multiplier)
      self._global_state = None
      self._num_parameters = num_parameters
      self._was_dp_gradients_called = False
      self._batch_idx = 0
      self.samples_cond = {}

    def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
      """DP version of superclass method."""
      print("compute the gradient")
      is_considered = [x.name.startswith("Considered") for x in var_list]
    
      print(is_considered)
      self._was_dp_gradients_called = True
      # Precompute the noise locations
      if len(self.samples_cond) == 0:
        self.samples_cond = [tf.Variable(random_choice_cond(x, self._num_parameters)) for x in var_list]
      # Compute loss.
      if not callable(loss) and tape is None:
        raise ValueError('`tape` is required when a `Tensor` loss is passed.')
      tape = tape if tape is not None else tf.GradientTape()
      self.is_considered = is_considered
      if callable(loss):
        with tape:
          if not callable(var_list):
            tape.watch(var_list)

          if callable(loss):
            loss = loss()
          else:
            microbatch_losses = tf.reduce_mean(
                tf.reshape(loss, [self._num_microbatches, -1]), axis=1)

          if callable(var_list):
            var_list = var_list()
      else:
        with tape:
            microbatch_losses = tf.reduce_mean(
              tf.reshape(loss, [self._num_microbatches, -1]), axis=1)
      var_list = tf.nest.flatten(var_list)
      
      # Compute the per-microbatch losses using helpful jacobian method.
      with tf.keras.backend.name_scope(self._name + '/gradients'):
        #tf.print(microbatch_losses.shape)
        jacobian = tape.jacobian(microbatch_losses, var_list)
        # the size of microbatch_losses is [num_microbatches, batch_size/num_microbatches]
        # the size of resulting jacobian will be [num_microbatches, batch_size/num_microbatches, num_parameters].
        #print("unstack:", tf.unstack(microbatch_losses))
        #t = [tape.gradient(one_loss, var_list) for one_loss in tf.unstack(microbatch_losses)]
        #print("t:", t)
        #jacobian = tf.stack(t)
        #tape.gradient(microbatch_losses, var_list)
        print(jacobian)
        # Clip gradients to given l2_norm_clip.
        def clip_gradients(g):
          return tf.clip_by_global_norm(g, self._l2_norm_clip)[0]

        clipped_gradients = tf.map_fn(clip_gradients, jacobian)
        print(clipped_gradients)
        
        def reduce_noise_normalize_batch(self, g, is_considered, layer_index):
          # Sum gradients over all microbatches.
          summed_gradient = tf.reduce_sum(g, axis=0)

          # Sample the indexes
          sampled_cond = self.samples_cond[layer_index]

          # Add noise to summed gradients.
          noise_stddev = self._l2_norm_clip * self._noise_multiplier
          noise = tf.random.normal(
              tf.shape(input=summed_gradient), stddev=noise_stddev)
          noised_gradient = tf.add(summed_gradient, noise)
          #tf.print(layer_index)
          #if layer_index == 2:
          #    tf.print(sampled_cond)
          #tf.print("num of noise:", tf.math.reduce_sum(tf.cast(tf.math.logical_or(sampled_cond, tf.math.logical_not(is_considered)), tf.int32)))
          fixed_gradient = tf.where(tf.math.logical_and(sampled_cond, is_considered), noised_gradient, summed_gradient)
          # Normalize by number of microbatches and return.
          return tf.truediv(fixed_gradient, self._num_microbatches)
            

        final_gradients = [reduce_noise_normalize_batch(self, clipped_gradients[i], is_considered[i], i) for i in range(len(clipped_gradients))]
        print(final_gradients)
        
        
        self.clipped_gradients = clipped_gradients
        self.final_gradients = final_gradients
      return list(zip(final_gradients, var_list))

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
      assert self._was_dp_gradients_called, (
          'Neither _compute_gradients() or get_gradients() on the '
          'differentially private optimizer was called. This means the '
          'training is not differentially private. It may be the case that '
          'you need to upgrade to TF 2.4 or higher to use this particular '
          'optimizer.')
      print(grads_and_vars)
      return super(FixedDPOptimizerClass,
                   self).apply_gradients(grads_and_vars, global_step, name)

  return FixedDPOptimizerClass

FixedDPKerasSGDOptimizer = make_fixed_keras_optimizer_class(tf.keras.optimizers.SGD)

def compute_epsilon(steps, sampling_probability, noise_multiplier):
  """Computes epsilon value for given hyperparameters."""
  if noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to 1e-5 because MNIST has 60000 training points.
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.bias
    
    
class TiedBiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(TiedBiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[-1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.bias