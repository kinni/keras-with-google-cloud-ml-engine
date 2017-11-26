
JOB_NAME="movie_sentiment_"$(date +%s)
JOB_DIR="gs://devfest2017-movie-sentiment/sentiment_keras_cloud_ml"
GCS_DATA_FILE="gs://devfest2017-movie-sentiment/data.csv"

gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.2 \
                                    --job-dir $JOB_DIR \
                                    --package-path trainer \
                                    --module-name trainer.task \
                                    --region us-central1 \
                                    -- \
                                    --data-file $GCS_DATA_FILE 