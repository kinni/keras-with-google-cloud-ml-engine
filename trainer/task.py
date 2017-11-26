# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This code implements a Feed forward neural network using Keras API."""

import argparse
import glob
import json
import os
import time

import keras
from keras.models import load_model
import model
from tensorflow.python.lib.io import file_io

INPUT_SIZE = 55
CLASS_SIZE = 2

FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
SENTIMENT_MODEL = 'sentiment.hdf5'


def dispatch(data_file,
             job_dir,
             num_epochs):

    timestamp = str(time.time())
    job_dir = job_dir + "/run" + timestamp

    x_train, y_train, maxlen, nb_chars, char_indices, indices_char = model.get_training_data(
        data_file[0])

    embedding_dims = 128
    filters = 250
    kernel_size = 3
    hidden_dims = 8
    batch_size = 32
    dropout_rate = 0.3
    epochs = num_epochs

    sentiment_model = model.model_fn(nb_chars=nb_chars,
                                     maxlen=maxlen,
                                     embedding_dims=embedding_dims,
                                     hidden_dims=hidden_dims,
                                     dropout_rate=dropout_rate,
                                     filters=filters,
                                     kernel_size=kernel_size)

    try:
        os.makedirs(job_dir)
    except:
        pass

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    checkpoint_path = FILE_PATH
    if not job_dir.startswith("gs://"):
        checkpoint_path = os.path.join(job_dir, checkpoint_path)

    # Model checkpoint callback
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')

    timestamp = str(time.time())

    # Tensorboard logs callback
    tblog = keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'),
        write_graph=True,
        embeddings_freq=0)

    callbacks = [checkpoint, tblog]

    sentiment_model = model.compile_model(sentiment_model)
    sentiment_model.fit(x_train, y_train,
                        epochs=epochs,
                        validation_split=0.3,
                        batch_size=batch_size,
                        callbacks=callbacks)

    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    if job_dir.startswith("gs://"):
        sentiment_model.save(SENTIMENT_MODEL)
        copy_file_to_gcs(job_dir, SENTIMENT_MODEL)
    else:
        sentiment_model.save(os.path.join(job_dir, SENTIMENT_MODEL))

    # Convert the Keras model to TensorFlow SavedModel
    model.to_savedmodel(sentiment_model, os.path.join(job_dir, 'export'))


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file',
                        required=True,
                        type=str,
                        help='Data file local or GCS', nargs='+')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Maximum number of epochs on which to train')
    parse_args, unknown = parser.parse_known_args()

    dispatch(**parse_args.__dict__)
