#!/usr/bin/env bash
#
# download_data.sh
#
# Helper script for preparing raw airline tweet datasets.
#
# This script is intentionally lightweight: in many research settings,
# raw data is either:
#   - confidential / licensed, or
#   - downloaded manually from a platform (e.g., Kaggle, airline APIs),
# so we do NOT hard-code any external URLs here.
#
# Instead, this script:
#   1) Creates the expected data/raw/ directory structure.
#   2) Shows clear instructions about which files should be placed there.
#
# If you have internal URLs or storage buckets, you can extend this
# script with curl/wget/gsutil/aws cli commands as needed.
#

set -e

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
RAW_DIR="${PROJECT_ROOT}/data/raw"

echo ">>> Ensuring raw data directory exists at: ${RAW_DIR}"
mkdir -p "${RAW_DIR}"

cat <<EOF

========================================================
Airline Tweet Sentiment – Raw Data Setup
========================================================

Please place your raw CSV files in:

  ${RAW_DIR}

The expected filenames and their meaning are configured in:

  config/config.yaml

Example configuration:

  datasets:
    airline_us:
      filename: "airline_tweets_us.csv"
    airline_global:
      filename: "airline_tweets_global.csv"
    airline_merged:
      filename: "airline_tweets_merged.csv"

Each CSV should contain at least:
  - A text column   (e.g., "text", "tweet_text", "content")
  - A label column  (e.g., "airline_sentiment", "sentiment", "label")

Once the files are in place, run:

  python scripts/prepare_datasets.py

to generate cleaned, stratified train/val/test splits under:

  data/processed/<dataset_name>/

If your data lives on a remote server, cloud bucket, or Kaggle,
you can now edit this script to add commands like:

  # Example (Kaggle) – placeholder
  # kaggle datasets download -d <owner>/<dataset> -p "${RAW_DIR}" --unzip

========================================================
EOF

echo ">>> Raw data directory is ready. See instructions above."
