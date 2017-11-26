# Build A Chinese Movie Sentiment Classifier with Keras and Google Cloud ML Engine

This is source project for my sharing on GDG HK DevFest 2017. This project will show how we can run Keras cloud on both local development machine and Google Cloud ML Engine. A simple character level CNN model is built using Keras.

## Slides
You may want to check out my slides at https://goo.gl/KVm52V.

## Prerequisites
 * Install [gcloud](https://cloud.google.com/sdk/gcloud/)
 * Install the python dependencies. `pip install --upgrade -r requirements.txt`

## Scripts
Train, TensorBoard and Predict
* gcloud-train-*.sh
    * Scripts for training the model locally or on Google Cloud ML Engine
* gcloud-tensorboard-*.sh
    * Scripts for visualizing the training process locally or on Google Cloud ML Engine
* gcloud-predict-*.sh
    * Scripts for making prediction locally or on Google Cloud ML Engine

## Billing
If you concern how much money will be burned using Google Cloud ML Engine, the answer is less than US$1 for this project. The model is simple and we don't have much dataset. So it is very safe to play around and have fun.

## For you to try out
* Change the model `hidden_dims` to larger (64) or even larger value
* Use one-hot encoding instead of the embedding layer
* Change `CNN` to `LSTM` model

## Reference
https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census/keras

Changes
* Predicting income => predicting Movie Sentiment
* Added multiple runs support for TensorBoard
* Fixed for  "You must feed a value for placeholder tensor `keras_learning_phase` with dtype uint8"
* Packed commands into shell scripts

