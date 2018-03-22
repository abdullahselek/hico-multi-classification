#!/usr/bin/env python

from __future__ import division

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
