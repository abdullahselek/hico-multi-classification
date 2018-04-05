#!/usr/bin/env python

'''A Benchmark for Recognizing Human-Object Interactions in Images with Pyhton.'''

from __future__ import absolute_import

__author__       = 'Abdullah Selek'
__email__        = 'abdullahselek@gmail.com'
__copyright__    = 'Copyright (c) 2018 Abdullah Selek'
__license__      = 'MIT License'
__version__      = '0.1'
__url__          = 'https://github.com/abdullahselek/hico-multi-classification'
__download_url__ = 'https://github.com/abdullahselek/hico-multi-classification'
__description__  = 'A Benchmark for Recognizing Human-Object Interactions in Images with Pyhton.'

from hico_multi_classification.hico_processor import (
    HicoProcessor
)

from hico_multi_classification.tfrecord_converter import (
    TFRecordConverter   
)
