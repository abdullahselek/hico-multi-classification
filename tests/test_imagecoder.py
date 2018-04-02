#!/usr/bin/env python

import unittest
import numpy as np

from hico_multi_classification.tfrecord_converter import ImageCoder

class ImageCoderTest(unittest.TestCase):

    def setUp(self):
        self._image_coder = ImageCoder()

    def test_init(self):
        self.assertIsNotNone(self._image_coder)

    def test_png_to_jpeg(self):
        with open('testdata/python_logo.png', 'rb') as imageFile:
            file = imageFile.read()
            array = np.array(file)
            jpeg = self._image_coder.png_to_jpeg(array)
            self.assertIsNotNone(jpeg)

    def test_cmyk_to_rgb(self):
        with open('testdata/python_logo.jpg', 'rb') as imageFile:
            file = imageFile.read()
            array = np.array(file)
            rgb = self._image_coder.cmyk_to_rgb(array)
            self.assertIsNotNone(rgb)

    def test_decode_jpeg(self):
        with open('testdata/python_logo.jpg', 'rb') as imageFile:
            file = imageFile.read()
            array = np.array(file)
            rgb = self._image_coder.decode_jpeg(array)
            self.assertIsNotNone(rgb)
