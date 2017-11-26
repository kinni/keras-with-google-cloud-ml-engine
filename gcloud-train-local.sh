JOB_DIR="sentiment_keras_local"
GCS_DATA_FILE="./data/data.csv"

gcloud ml-engine local train --package-path trainer \
                             --module-name trainer.task \
                             -- \
                             --data-file $GCS_DATA_FILE \
                             --job-dir $JOB_DIR \