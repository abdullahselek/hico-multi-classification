#!/usr/bin/env python

from __future__ import division
import os
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

    def __convert_to_example(self,
                             image_data,
                             filename,
                             label,
                             label_text,
                             height,
                             width):
        """Build an Example proto for an example.

        Args:
          image_data (str):
            JPEG encoding of RGB image
          filename (str):
            filename of an image
          label (list of floats):
            identifier for the ground truth
          label_text (list of str):
            e.g. ['airplane board', 'airplane ride']
          height (integer):
            image height in pixels
          width (integer):
            image width in pixels
        Returns:
          Example proto
        """

        colorspace = 'RGB'
        channels = 3
        image_format = 'JPEG'

        obj = []
        verb = []

        for text in label_text:
            assert len(text) == 2
            obj.append(text[0])
            verb.append(text[1])

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': self.__int64_feature(height),
            'image/width': self.__int64_feature(width),
            'image/colorspace': self.__bytes_feature(colorspace),
            'image/channels': self.__int64_feature(channels),
            'image/format': self.__bytes_feature(image_format),
            'image/filename': self.__bytes_feature(filename),
            'image/encoded': self.__bytes_feature(image_data),
            'image/class/label': self.__bytes_feature(label),
            'image/class/object': self.__bytes_feature(obj),
            'image/class/verb': self.__bytes_feature(verb)
            }))
        return example

    def __process_image(self,
                        data_dir,
                        filename,
                        coder):
        """Process a single image file.

        Args:
          data_dir (str):
            Root directory of images
          filename (str):
            Filename of an image file.
          coder (ImageCoder):
            Instance of ImageCoder to provide TensorFlow image coding utils.
        Returns:
          image_data (str):
            JPEG encoding of RGB image.
          height (int):
            Image height in pixels.
          width (int):
            Image width in pixels.
        """
        # Read the image file.
        file_path = os.path.join(data_dir, filename)
        image_data = tf.gfile.FastGFile(file_path, 'r').read()

        # Decode the RGB JPEG.
        image = coder.decode_jpeg(image_data)

        # Check that image converted to RGB
        assert len(image.shape) == 3
        height = image.shape[0]
        width = image.shape[1]
        assert image.shape[2] == 3

        return image_data, height, width

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._tfsession = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._tfsession.run(self._png_to_jpeg,
                                   feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._tfsession.run(self._cmyk_to_rgb,
                                   feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._tfsession.run(self._decode_jpeg,
                                    feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
