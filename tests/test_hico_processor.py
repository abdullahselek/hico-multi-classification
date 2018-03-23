#!/usr/bin/env python

import unittest
from hico_multi_classification.hico_processor import HicoProcessor

class HicoProcessorTest(unittest.TestCase):

    def setUp(self):
        processor = HicoProcessor('testdata/hico_data/hico_20150920', 'anno.mat')
        self._processor = processor

    def test_init(self):
        self.assertIsNotNone(self._processor)
