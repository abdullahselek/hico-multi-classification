#!/usr/bin/env python

from __future__ import division

import os
import scipy.io as sio

class HicoProcessor(object):
    """A processor which reads from mat file and
    prepare output labels.
    """

    def __init__(self, data_dir, anno_mat):
        """Returns a HicoProcessor instance.
        Args:
          data_dir (str):
            Directory path for hico data.
          anno_mat_dir (str):
            File name for annotation mat file.
        """

        self._data_dir = data_dir
        self._anno_mat = anno_mat

    def __load_mat(self, data_dir, anno_mat):
        """Read the labels from annotation mat file and return.
        Args:
          data_dir (str):
            Directory path for hico data.
          anno_mat_dir (str):
            File name for annotation mat file.
        """
        path = os.path.join(data_dir, anno_mat)
        return sio.loadmat(path)
