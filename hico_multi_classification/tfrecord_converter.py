#!/usr/bin/env python

from __future__ import division
import tensorflow as tf

class TFRecordConverter(object):
    """Class converts HICO data to Tensorflow records."""
    
    def __int64_feature(self, value):
        """Inserting int64 features into proto."""

        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
