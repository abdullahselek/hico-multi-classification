#!/usr/bin/env python

from __future__ import division
import tensorflow as tf

tf.app.flags.DEFINE_string('data_dir', './hico_data/',
						   'Data directory')
tf.app.flags.DEFINE_string('output_dir', './tfrecords',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 32,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('test_shards', 16,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')

# This file contains mapping from label to object-verb description.
# Assumes each line of the file looks like:
#
#   0 airplane board
#   28 bird release
#   39 boat row
#
# where each line corresponds to a unique mapping. Note that each line is
# formatted as <label> <object> <verb>.
tf.app.flags.DEFINE_string('label_text',
                           'label_text.txt',
                           'Label description file')

# This file contains training labels
# Assumes each line of the file looks like:
#
#   576 577 578
#
tf.app.flags.DEFINE_string('labels_train',
                           'labels_train.txt',
                           'Training label file')
tf.app.flags.DEFINE_string('labels_test',
						   'labels_test.txt',
						   'Test label file')
tf.app.flags.DEFINE_integer('num_classes',
                           600,
                           'Number of classes')

# This file contains training filenames
# Assumes each line of the file looks like:
#
#	HICO_train2015_00000036.jpg
#
tf.app.flags.DEFINE_string('filenames_train',
						   'filenames_train.txt',
						   'Training filenames file')
tf.app.flags.DEFINE_string('filenames_test',
						   'filenames_test.txt',
						   'Testing filenames file')

FLAGS = tf.app.flags.FLAGS

class TFRecordConverter(object):
    """Class converts HICO data to Tensorflow records."""
    
    def __int64_feature(self, value):
        """Inserting int64 features into proto."""

        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def __float_feature(self, value):
        """Inserting float features into proto."""

        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def __bytes_feature(self, value):
        """Inserting bytes features into proto."""

        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
