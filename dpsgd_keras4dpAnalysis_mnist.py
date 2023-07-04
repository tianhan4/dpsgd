# %%
# Experiment: 2021/03/14
# Usage: Dp analysis of mnist neural network.
# Detail: Check the notion notes in HE-DP project.

from dpsgd_keras_slow import *
import pickle

dpsgd = False # add dp noise or not 
learning_rate = 0.1
noise_multiplier = 4
l2_norm_clip = 3
batch_size = 500#512
epochs = 20
microbatch_size = 25
num_parameters = 0
privacy_budget = []
delta = 1e-5  # it is recommended to use delta~=1/dataset_size
model_dir = None

# %%
tf.__version__

# %%
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# %%
tf.config.list_physical_devices('GPU')

# %%

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# %%
def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
  test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.

  return train_data, train_labels, test_data, test_labels

# %%
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

# %%
#tf.clip_by_global_norm([tf.Variable([[2,3],[4,5]],shape=[2,2],dtype=tf.float32)], 4.4)[0][0]

# %%
# Models 
def build_models(noise_layer_name):
    if noise_layer_name =="mnist+untied_bias+noise_input":
        model = tf.keras.Sequential([
            GaussianNoiseLayer(l2_norm_clip, noise_multiplier, input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(10, activation=None, use_bias=False),
            BiasLayer()
        ])        
    elif noise_layer_name == "mnist+tied_bias+noise_input":
        model = tf.keras.Sequential([
            GaussianNoiseLayer(l2_norm_clip, noise_multiplier, input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None, use_bias=False),
            TiedBiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False),
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
    elif noise_layer_name == "mnist+untied_bias":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(10, activation=None, use_bias=False),
            BiasLayer()
        ])
    elif noise_layer_name == "mnist+tied_bias":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False),
            TiedBiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False),
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
    elif noise_layer_name == "mnist":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=None, use_bias=False),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(10, activation=None, use_bias=False),
            BiasLayer()
        ])  
    elif noise_layer_name == "sphinx+mnist+untied_bias":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False, name="Considered1"),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False, name="Considered2"),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=None, use_bias=False, name="Considered3"),
            BiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(10, activation=None, use_bias=False, name="Considered4"),
            BiasLayer()
        ])
    elif noise_layer_name == "sphinx+mnist+tied_bias":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False, name="Considered1"),
            TiedBiasLayer(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False, name="Considered2"),
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
    elif noise_layer_name == "ALLnoise+mnist+untied_bias":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False, name="Considered1"),
            BiasLayer(name="Considered2"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False, name="Considered3"),
            BiasLayer(name="Considered4"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=None, use_bias=False, name="Considered5"),
            BiasLayer(name="Considered6"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(10, activation=None, use_bias=False, name="Considered7"),
            BiasLayer(name="Considered8")
        ])
    elif noise_layer_name == "ALLnoise+mnist+tied_bias":
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False, name="Considered1"),
            TiedBiasLayer(name="Considered7"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2,2), 2),
            tf.keras.layers.Conv2D(16, 5,
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 input_shape=(28, 28, 1), use_bias=False, name="Considered2"),
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
    elif noise_layer_name == "cifar10":
        pass
    else:
        model = None
    return model

# %%
build_models("sphinx+mnist+untied_bias").summary()

# %%
#def main_simple(unused_argv):
logging.set_verbosity(logging.INFO)
if dpsgd and batch_size % microbatch_size != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

# Load training and test data.
train_data, train_labels, test_data, test_labels = load_mnist()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=(epochs * train_data.shape[0]) // microbatch_size // 1000,
    decay_rate=0.998)


lr_schedule_no_dp = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=(epochs * train_data.shape[0]) // batch_size // 1000,
    decay_rate=0.998)

# %%
# check the training accuracy
dpsgd = True
parameter_list = {}
accuracies = {}
#for model_type in ["mnist", "mnist+tied_bias", "mnist+untied_bias"]:
for model_type in ["sphinx+mnist+tied_bias"]: #, "ALLnoise+mnist+tied_bias", "mnist+tied_bias+noise_input"]:
    file_name = model_type + "_accuracy"
    if model_type == "mnist+untied_bias+noise_input":
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
            learning_rate=lr_schedule)
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule_no_dp)
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

print("final accuracy history:", accuracies)

# save the accuracies, file name is the model name plus the noise level and clip norm C
with open("./dp_data/accuracies" + str(noise_multiplier) + "_" + str(l2_norm_clip), "wb") as f:
    pickle.dump(accuracies, f)