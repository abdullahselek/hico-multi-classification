#!/bin/bash

if [[ "$1" = "hico_data" ]]; then
    mkdir "$1"
    echo "Downlading from http://napoli18.eecs.umich.edu/public_html/data/hico_20150920.tar.gz"
    curl http://napoli18.eecs.umich.edu/public_html/data/hico_20150920.tar.gz --output hico_20150920.tar.gz
    tar -zxvf hico_20150920.tar.gz --directory "$1"
    rm hico_20150920.tar.gz
    echo "Hico data downloaded"
    exit 1
fi

if [[ "$1" = "inception" ]]; then
    mkdir "$1"
    echo "Downlading from http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
    curl http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz --output inception_v3_2016_08_28.tar.gz
    tar -zxvf inception_v3_2016_08_28.tar.gz --directory "$1"
    rm inception_v3_2016_08_28.tar.gz
    echo "Inception downladed"
    exit 1
fi
