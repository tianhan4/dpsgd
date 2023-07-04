
# Experiment: 2021/03/14
# Usage: Dp analysis of mnist neural network.
# Detail: Check the notion notes in HE-DP project.

from dpsgd_keras_slow import *

dpsgd = False # add dp noise or not 
learning_rate = 0.1
noise_multiplier = 1
l2_norm_clip = 3
batch_size = 1024
epochs = 40
microbatch_size = 16
num_parameters = 0
privacy_budget = []
delta = 1e-5  # it is recommended to use delta~=1/dataset_size
model_dir = None


# In[5]:


import tensorflow as tf
tf.__version__


# In[ ]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7,8'


# In[ ]:


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# In[ ]:


tf.config.list_physical_devices('GPU')


# In[ ]:



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[ ]:


train, test = tf.keras.datasets.cifar10.load_data()


# In[ ]:


def load_cifar10():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.cifar10.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_data = train_data.reshape((train_data.shape[0], 32, 32, 3))
  test_data = test_data.reshape((test_data.shape[0], 32, 32, 3))

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.

  return train_data, train_labels, test_data, test_labels


# In[ ]:


# Perturnbing the input dataset, and collect the accuracy-step
class GaussianNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, l2_norm_clip, noise_multiplier, *args, **kwargs):
        super(GaussianNoiseLayer, self).__init__(*args, **kwargs)
        self._l2_norm_clip = l2_norm_clip
        self._noise_multiplier = noise_multiplier

    def build(self, input_shape):
        pass
    
    def call(self, x):
        # Clip gradients to given l2_norm_clip.
        def clip_features(x):
            return tf.clip_by_global_norm([x], self._l2_norm_clip)[0][0]

        clipped_features = tf.map_fn(clip_features, x)
        
        # Add noise to summed gradients.
        noise_stddev = self._l2_norm_clip * self._noise_multiplier
        noise = tf.random.normal(tf.shape(input=clipped_features), stddev=noise_stddev)
        return clipped_features + noise


# In[ ]:


#tf.clip_by_global_norm([tf.Variable([[2,3],[4,5]],shape=[2,2],dtype=tf.float32)], 4.4)[0][0]


# In[ ]:


# Models 



def build_models(noise_layer_name):
    if noise_layer_name =="cifar10+untied_bias+noise_input":
        model = tf.keras.Sequential([
            GaussianNoiseLayer(l2_norm_clip, noise_multiplier, input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None,
                                 input_shape=(32, 32, 3), use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 1,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(16, 1,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=None, use_bias=False),
            BiasLayer()
        ])
    elif noise_layer_name == "cifar10+tied_bias+noise_input":
        model = tf.keras.Sequential([
            GaussianNoiseLayer(l2_norm_clip, noise_multiplier, input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None,
                                 input_shape=(32, 32, 3), use_bias=False),
            TiedBiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            TiedBiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            TiedBiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            TiedBiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            TiedBiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 1,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            TiedBiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(16, 1,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            TiedBiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=None, use_bias=False),
            BiasLayer()
        ])
    elif noise_layer_name == "cifar10+untied_bias":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None,
                                 input_shape=(32, 32, 3), use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 1,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(16, 1,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=None, use_bias=False),
            BiasLayer()
        ])
    
    
    elif noise_layer_name == "cifar10+tied_bias":
            model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(32, 32, 3), use_bias=False),
            TiedBiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None, use_bias=False),
            TiedBiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(10, activation=None, use_bias=False),
            BiasLayer()
        ])
    elif noise_layer_name == "cifar10":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None,
                                 input_shape=(32, 32, 3), use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 1,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(16, 1,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=None, use_bias=False),
            BiasLayer()
        ])  
    elif noise_layer_name == "sphinx+cifar10+untied_bias":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None,
                                 input_shape=(32, 32, 3), use_bias=False, name="Considered1"),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False, name="Considered2"),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False, name="Considered3"),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False, name="Considered4"),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False, name="Considered5"),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 1,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False, name="Considered6"),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(16, 1,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False, name="Considered7"),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=None, use_bias=False, name="Considered8"),
            BiasLayer()
            ])
    elif noise_layer_name == "sphinx+cifar10+tied_bias":
        model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 5,
                             strides=1,
                             padding='valid',
                             activation=None,
                             input_shape=(32, 32, 3), use_bias=False, name="Considered1"),
        TiedBiasLayer(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D((2,2), 2),
        tf.keras.layers.Conv2D(16, 5,
                             strides=1,
                             padding='valid',
                             activation=None, use_bias=False, name="Considered2"),
        TiedBiasLayer(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D((2,2), 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation=None, use_bias=False, name="Considered3"),
        BiasLayer(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(10, activation=None, use_bias=False, name="Considered4"),
        BiasLayer()
    ])
    elif noise_layer_name == "ALLnoise+cifar10+untied_bias":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None,
                                 input_shape=(32, 32, 3), use_bias=False, name="Considered1"),
            BiasLayer(name="Considered9"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False, name="Considered2"),
            BiasLayer(name="Considered10"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False, name="Considered3"),
            BiasLayer(name="Considered11"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False, name="Considered4"),
            BiasLayer(name="Considered12"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(64, 3,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False, name="Considered5"),
            BiasLayer(name="Considered13"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 1,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False, name="Considered6"),
            BiasLayer(name="Considered14"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(16, 1,
                                 strides=1,
                                 padding='same',
                                 activation=None, use_bias=False, name="Considered7"),
            BiasLayer(name="Considered15"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=None, use_bias=False, name="Considered8"),
            BiasLayer(name="Considered16")
        ])
    elif noise_layer_name == "ALLnoise+cifar10+tied_bias":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(32, 32, 3), use_bias=False, name="Considered1"),
            TiedBiasLayer(name="Considered7"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None, use_bias=False, name="Considered2"),
            TiedBiasLayer(name="Considered8"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=None, use_bias=False, name="Considered3"),
            BiasLayer(name="Considered4"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(10, activation=None, use_bias=False, name="Considered5"),
            BiasLayer(name="Considered6")
        ])
    else:
        model = None
    return model


# In[ ]:


build_models("cifar10+untied_bias+noise_input").summary()


# In[ ]:


build_models("cifar10+untied_bias").summary()


# In[ ]:


build_models("cifar10+tied_bias").summary()


# In[ ]:


build_models("cifar10").summary()


# In[ ]:


#def main_simple(unused_argv):
logging.set_verbosity(logging.INFO)
if dpsgd and batch_size % microbatch_size != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

# Load training and test data.
train_data, train_labels, test_data, test_labels = load_cifar10()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=(epochs * train_data.shape[0]) // microbatch_size // 1000,
    decay_rate=0.998)


# In[ ]:


# check the training accuracy
origin_dpsgd = dpsgd
parameter_list = {}
accuracies = {}
#for model_type in ["mnist", "mnist+tied_bias", "mnist+untied_bias"]:
for model_type in ["sphinx+cifar10+tied_bias", "ALLnoise+cifar10+tied_bias","cifar10+tied_bias"]:
    file_name = model_type + "_accuracy"
    if model_type == "cifar10+tied_bias+noise_input":
        dpsgd = False
    else:
        dpsgd = origin_dpsgd
    model = build_models(model_type)
    if dpsgd:
        optimizer = FixedDPKerasSGDOptimizer(
            batch_size = batch_size,
            num_parameters = num_parameters,
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            var_list = model.trainable_variables,
            #num_microbatches=batch_size//microbatch_size,
            microbatch_size = microbatch_size,
            learning_rate=learning_rate)
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Compile model with Keras
    checkpoint_filepath = "./dp_data/"+file_name
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_accuracy',
      mode='max',
      save_best_only=True)
    #early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    ##    monitor='val_loss', min_delta=0, patience=5, verbose=0,
    #    mode='auto', baseline=None, restore_best_weights=True
    #)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # Train model with Keras
    if dpsgd:
        history = model.fit(train_data, train_labels,
                epochs=epochs,
                validation_data=(test_data, test_labels),
                batch_size=microbatch_size, 
                callbacks = [model_checkpoint_callback], workers=1) # , early_stopping_callback
    else:
        history = model.fit(train_data, train_labels,
                epochs=epochs,
                validation_data=(test_data, test_labels),
                batch_size=batch_size, 
                callbacks = [model_checkpoint_callback], workers=1) # , early_stopping_callback
    
    #evaluated_result = model.evaluate(
    #    x=test_data, y=test_labels, batch_size=None, verbose=1, sample_weight=None, steps=None,
    #    callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
    #    return_dict=True)
    if model_type not in accuracies.keys():
        accuracies[model_type] = []
    accuracies[model_type].append(history)
    #accuracies[model_type].append(evaluated_result["accuracy"])

# Compute the privacy budget expended.
#if dpsgd:
#    eps = compute_epsilon(epochs * 60000 // batch_size)
#    print('For delta=1e-5, the current epsilon is: %.2f' % eps)
#else:
#    print('Trained with vanilla non-private SGD optimizer')


# In[ ]:


accuracies


# In[ ]:



processed_accuracies = dict()
for key in accuracies.keys():   
    processed_accuracies[key] = accuracies[key][0].history
    
import pickle
file = open("dp_data/results/cifar10_accuracies_%.1f_%.1f_%.1f_%d"%(learning_rate, noise_multiplier, l2_norm_clip, batch_size),"wb")
pickle.dump(processed_accuracies, file)
file.close()


# In[ ]:


assert(False)


# In[ ]:


processed_accuracies


# ## Perturbed Label : skip
# 

# In[ ]:


def clip_noise(x, clip, noise_multiplier):
    def clip_features(x):
        return tf.clip_by_global_norm([x], clip)[0][0]
    clipped_features = tf.map_fn(clip_features, x)
    # Add noise to summed gradients.
    noise_stddev = clip * noise_multiplier
    noise = tf.random.normal(tf.shape(input=clipped_features), stddev=noise_stddev)
    return clipped_features + noise

noisy_train_data = tf.map_fn(lambda x: clip_noise(x, l2_norm_clip, noise_multiplier), train_data)


# In[ ]:


accuracies


# In[ ]:


processed_accuracies = dict()
for key in accuracies.keys():   
    processed_accuracies[key] = accuracies[key][0].history


# In[ ]:


processed_accuracies


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(processed_accuracies['sphinx+mnist+untied_bias']['accuracy'])
plt.plot(processed_accuracies['sphinx+mnist+untied_bias']['val_accuracy'])
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


import pickle
file = open("dp_data/results/accuracies_%.1f_%.1f_%.1f_%d"%(learning_rate, noise_multiplier, l2_norm_clip, batch_size),"wb")
pickle.dump(processed_accuracies, file)
file.close()


# In[ ]:


assert(False)


# ## get the step-epsilon curve

# In[ ]:


import numpy as np
import pickle 

try:
    with open("dp_data/noise_epsilon_step") as f:
        epsilons = pickle.load(f)
except:
    training_data_size = 60000
    max_step = 25000
    epsilons = [[] for i in range(10)]
    for noise_multiplier in range(0, 10, 1):
        print("noise_multiplier:", noise_multiplier)
        for step in range(1, max_step, 10):
            epsilons[noise_multiplier].append(compute_epsilon(step, batch_size/training_data_size, noise_multiplier))
    file = open("dp_data/noise_epsilon_step_%d"%batch_size,"wb")
    pickle.dump(epsilons, file)
    file.close()


# ## Tied Bias

# In[ ]:


# For tied bias
#for model_type in ["mnist", "mnist+tied_bias", "mnist+untied_bias"]:
for model_type in ["sphinx+mnist+tied_bias", "ALLnoise+mnist+tied_bias","mnist+tied_bias"]:
    file_name = model_type + "_accuracy"
    if model_type == "mnist+tied_bias+noise_input":
        dpsgd = False
    else:
        dpsgd = True
    model = build_models(model_type)
    if dpsgd:
        optimizer = FixedDPKerasSGDOptimizer(
            batch_size = batch_size,
            num_parameters = num_parameters,
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            var_list = model.trainable_variables,
            #num_microbatches=batch_size//microbatch_size,
            microbatch_size = microbatch_size,
            learning_rate=learning_rate)
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Compile model with Keras
    checkpoint_filepath = "./dp_data/"+file_name
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_accuracy',
      mode='max',
      save_best_only=True)
    #early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    ##    monitor='val_loss', min_delta=0, patience=5, verbose=0,
    #    mode='auto', baseline=None, restore_best_weights=True
    #)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # Train model with Keras
    if dpsgd:
        history = model.fit(train_data, train_labels,
                epochs=epochs,
                validation_data=(test_data, test_labels),
                batch_size=microbatch_size, 
                callbacks = [model_checkpoint_callback], workers=1) # , early_stopping_callback
    else:
        history = model.fit(train_data, train_labels,
                epochs=epochs,
                validation_data=(test_data, test_labels),
                batch_size=batch_size, 
                callbacks = [model_checkpoint_callback], workers=1) # , early_stopping_callback
    
    #evaluated_result = model.evaluate(
    #    x=test_data, y=test_labels, batch_size=None, verbose=1, sample_weight=None, steps=None,
    #    callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
    #    return_dict=True)
    if model_type not in accuracies.keys():
        accuracies[model_type] = []
    accuracies[model_type].append(history)
    #accuracies[model_type].append(evaluated_result["accuracy"])

# Compute the privacy budget expended.
#if dpsgd:
#    eps = compute_epsilon(epochs * 60000 // batch_size)
#    print('For delta=1e-5, the current epsilon is: %.2f' % eps)
#else:
#    print('Trained with vanilla non-private SGD optimizer')


# In[ ]:


accuracies


# In[ ]:


train_data


# In[ ]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# Using set_dashes() to modify dashing of an existing line
for key in parameter_list.keys():
    line1, = ax.plot(parameter_list[key],accuracies[key],  label=key)
    line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

ax.legend()
plt.show()


import pickle
result = open("result","wb")
pickle.dump(accuracies, result)
pickle.dump(parameter_list, result)
result.close()


# In[ ]:


result = open("result", "rb")
accuracies = pickle.load(result)
parameter_list = pickle.load(result)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


len(num)


# In[ ]:


parameter_list


# In[ ]:





# In[ ]:



fig, ax = plt.subplots()

# Using set_dashes() to modify dashing of an existing line
line1, = ax.plot(parameter_list["linear"],accuracies["linear"],  label='Linear')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

line2, = ax.plot(parameter_list["bias"],  accuracies["bias"], label='Bias')
line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

ax.legend()
plt.show()






