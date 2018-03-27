#!/usr/bin/env python

import os.path
import unittest
from hico_multi_classification.hico_processor import HicoProcessor

class HicoProcessorTest(unittest.TestCase):

    def setUp(self):
        processor = HicoProcessor('testdata/hico_data/hico_20150920', 'anno.mat')
        self._processor = processor

    def test_init(self):
        self.assertIsNotNone(self._processor)

    def test_process(self):
        self._processor.process()
        self.assertTrue(os.path.isfile(os.path.join(self._processor._data_dir, 'labels_train.txt')))
        self.assertTrue(os.path.isfile(os.path.join(self._processor._data_dir, 'labels_test.txt')))
        self.assertTrue(os.path.isfile(os.path.join(self._processor._data_dir, 'label_text.txt')))
        self.assertTrue(os.path.isfile(os.path.join(self._processor._data_dir, 'filenames_train.txt')))
        self.assertTrue(os.path.isfile(os.path.join(self._processor._data_dir, 'filenames_test.txt')))
