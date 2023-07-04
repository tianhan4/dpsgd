
"""Training a CNN on MNIST with Keras and the DP SGD optimizer.
Slow implementation allowing large batch size: using input-size=num_microbatch outside, B-batch inside.
Usage: input batch size also, which is the real update frequency."""
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

tf.config.threading.set_intra_op_parallelism_threads(1)

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
        batch_size,
        var_list,
        microbatch_size= 1,
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
      assert(batch_size % microbatch_size == 0)
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier
      self._aggregate_gradients = [tf.Variable(tf.zeros_like(grad)) for grad in var_list]
      self._dp_sum_query = gaussian_query.GaussianSumQuery(
          l2_norm_clip, l2_norm_clip * noise_multiplier)
      self._global_state = None
      self._num_parameters = num_parameters
      self._was_dp_gradients_called = False
      self._batch_idx = tf.Variable(0)
      self._batch_size = tf.Variable(batch_size)
      self._microbatch_size = microbatch_size
      self.samples_cond = {}

    @tf.function
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
          if callable(var_list):
            var_list = var_list()
      var_list = tf.nest.flatten(var_list)

      # Compute the per-microbatch losses using helpful jacobian method.
      with tf.keras.backend.name_scope(self._name + '/gradients'):
        #tf.print(microbatch_losses.shape)
        jacobian = tape.jacobian(loss, var_list)
        tf.print("jacobian shape:", len(jacobian), jacobian[0].shape)
        
        # map_fn also supports functions with multi-arity inputs and outputs:

        # If elems is a tuple (or nested structure) of tensors, then those tensors must all have the same outer-dimension size (num_elems); and fn is used to transform each tuple (or structure) of corresponding slices from elems. E.g., if elems is a tuple (t1, t2, t3), then fn is used to transform each tuple of slices (t1[i], t2[i], t3[i]) (where 0 <= i < num_elems).

        # If fn returns a tuple (or nested structure) of tensors, then the result is formed by stacking corresponding elements from those structures.
        # Clip gradients to given l2_norm_clip.
        def clip_gradients(g):
          # print the dimension of g
          # tf.print("g shape:", len(g), g[0].shape)
          #tf.print("g:", g)
          #tf.print(tf.linalg.global_norm(g))
          considered_g = [g[i] for i in range(len(g)) if self.is_considered[i]]
          # calculate the global norm of consider_g
          div_scale = tf.linalg.global_norm(considered_g)/self._l2_norm_clip
          if div_scale > 1:
            return [grad/div_scale for grad in g]
          else:
            return g

        clipped_gradients = tf.map_fn(clip_gradients, jacobian)
        # print("clipped_gradients:", clipped_gradients)
        
        final_gradients = [tf.reduce_sum(clipped_gradients[i], axis=0) for i in range(len(clipped_gradients))]
        # print(final_gradients)
        
        #self.clipped_gradients = clipped_gradients
        #self.final_gradients = final_gradients
        _aggregate_gradients = self._aggregate_gradients.copy()
        #tf.print(_aggregate_gradients)
        _batch_idx = self._batch_idx
        _batch_size = self._batch_size
        
        for i in range(len(final_gradients)):
          _aggregate_gradients[i] = tf.cond(_batch_idx == tf.constant(0), lambda :final_gradients[i], lambda :_aggregate_gradients[i] + final_gradients[i])
        
        for i in range(len(_aggregate_gradients)):
          self._aggregate_gradients[i].assign(_aggregate_gradients[i])
        
        _batch_idx = self._microbatch_size + _batch_idx 
        #tf.print(_batch_idx)
        def noise_normalize_batch(self, g, is_considered, layer_index):
          # Sample the indexes
          sampled_cond = self.samples_cond[layer_index]
          # Add noise to summed gradients.
          noise_stddev = self._l2_norm_clip * self._noise_multiplier
          noise = tf.random.normal(
            tf.shape(input=g), stddev=noise_stddev)
          noised_gradient = tf.add(g, noise)
          fixed_gradient = tf.where(tf.math.logical_and(sampled_cond, is_considered), noised_gradient, g)
          # Normalize by number of microbatches and return.
          return tf.truediv(fixed_gradient, tf.cast(_batch_size, dtype=tf.float32))
        
        noise_normalized_gradients = [noise_normalize_batch(self, _aggregate_gradients[i], self.is_considered[i], i) for i in range(len(_aggregate_gradients))]
        for i in range(len(final_gradients)):
          final_gradients[i] = tf.cond(_batch_idx >= _batch_size, lambda :noise_normalized_gradients[i], lambda :tf.zeros_like(final_gradients[i]))
        _batch_idx = tf.cond(_batch_idx >= _batch_size, lambda : tf.constant(0), lambda :_batch_idx + 0)
      #tf.print(final_gradients)
      self._batch_idx.assign(_batch_idx)
      return list(zip(final_gradients, var_list))

    #_aggregate_gradients: aggregated gradients until now.
    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
      assert self._was_dp_gradients_called, (
          'Neither _compute_gradients() or get_gradients() on the '
          'differentially private optimizer was called. This means the '
          'training is not differentially private. It may be the case that '
          'you need to upgrade to TF 2.4 or higher to use this particular '
          'optimizer.')
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