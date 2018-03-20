#!/bin/bash

if [[ "$1" = "hico" ]]; then
    mkdir "$1"
    echo "Downlading from http://napoli18.eecs.umich.edu/public_html/data/hico_20150920.tar.gz"
    curl http://napoli18.eecs.umich.edu/public_html/data/hico_20150920.tar.gz --output "$1"/hico_20150920.tar.gz
    cd "$1"
    tar -zxvf hico_20150920.tar.gz
    rm hico_20150920.tar.gz
    echo "Hico data downloaded"
    exit 1
fi

if [[ "$1" = "inception" ]]; then
    mkdir "$1"
    echo "Downlading from http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
    curl http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz --output "$1"/inception_v3_2016_08_28.tar.gz
    cd "$1"
    tar -zxvf inception_v3_2016_08_28.tar.gz
    rm inception_v3_2016_08_28.tar.gz
    echo "Inception downladed"
    exit 1
fi

