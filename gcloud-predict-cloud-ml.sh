MODEL="sentiment_model"
VERSION="v4"

gcloud ml-engine predict --model $MODEL --version $VERSION --json-instances sample.json