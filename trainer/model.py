import keras
import pandas as pd
import numpy as np
from keras import backend as K
from keras import layers, models
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.backend import relu, sigmoid

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter

_PAD_ = 0
_UNK_ = 1


def model_fn(nb_chars,
             maxlen,
             embedding_dims,
             hidden_dims,
             dropout_rate,
             filters=250,
             kernel_size=3):

    K.set_learning_phase(False) # Fix for  "You must feed a value for placeholder tensor 'keras_learning_phase' with dtype uint8"

    """Create a Keras Sequential model with layers."""
    model = models.Sequential()
    model.add(layers.Embedding(nb_chars,
                               embedding_dims,
                               input_length=maxlen))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv1D(filters,
                            kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1))
    model.add(layers.GlobalMaxPooling1D())

    model.add(layers.Dense(hidden_dims))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Activation('relu'))

    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    return model


def compile_model(model):
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'text': model.inputs[0]},
                                      outputs={'sentiment': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )
        builder.save()


def get_training_data(data_file):
    seed = 689

    dataset = pd.read_csv(tf.gfile.Open(data_file), encoding='utf-8',
                          names=['sample', 'label'], header=0)

    samples = dataset['sample'].str.lower()
    labels = dataset['label']

    maxlen = max(map(len, (x for x in samples)))
    chars = sorted(set("".join(samples)))
    char_size = len(chars) + 2

    # reserve 0 for padding and 1 for oov
    char_indices = dict((c, i + 2) for i, c in enumerate(chars))
    indices_char = dict((i + 2, c) for i, c in enumerate(chars))

    x_train = []
    y_train = []
    for sample in samples:
        x_train.append([char_indices[char] for char in sample])

    for label in labels:
        y_train.append(0 if label == 'negative' else 1)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
    y_train = np.array(y_train)

    # shuffle the indices
    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    return x_train, y_train, maxlen, char_size, char_indices, indices_char


def get_input_data(data_file, text):
    x_train, y_train, maxlen, char_size, char_indices, indices_char = get_training_data(
        data_file)

    X = [char_indices[char if char in char_indices else _UNK_]
         for char in list(text)]

    return sequence.pad_sequences([X], maxlen=maxlen, padding='post').tolist()
