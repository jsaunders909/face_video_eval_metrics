This repo is used for running several evaluation metrics on generated face videos.

Each metric is in a python file compute_X.py. If there are two inputs (e.g. comparing real to fake) you need to run using -r and -f flags pointing to **directories** of videos **with matching names**. If the metric requires only the generated videos (e.g. Sync metrics) use the -i flag with **directories**.