JOB_DIR="./sentiment_keras_cloud/run1510386200.3"

gcloud ml-engine local predict --model-dir=$JOB_DIR/export --json-instances sample.json