MODEL="sentiment_model"
MODEL_BINARIES="gs://devfest2017-movie-sentiment/sentiment_keras_cloud_ml/run1510390036.99/export/"
VERSION="v4"

gcloud ml-engine versions create $VERSION --model $MODEL --origin $MODEL_BINARIES --runtime-version 1.2