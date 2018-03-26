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

    def __prepare_output_labels(self,
                                anno_mat,
                                key,
                                file_name):
        """Prepare output labels to generate labels.
        Args:
          anno_mat (dict):
            Dictionary with variable names as keys, and loaded matrices as values.
          key (str):
            Key to training or test receive from dict.
        Returns:
          entry, labels_file:
            entry (i,j) is the annotation of training or test image j on action i.
            file path for label file.
        """

        anno_dict = anno_mat[key]
        labels_file = os.path.join(self._data_dir, file_name)
        return anno_dict, labels_file

    def __generate_labels_file(self,
                               anno_dict,
                               output_path):
        """Generate files for given label dicts.
        Args:
          anno_dict (dict):
            Dictionary with variable names as keys, and loaded matrices as values.
          output_path (str):
            File path to create labels.
        """
        labels = []
        for data in anno_dict.T:
            indices = [i for i,v in enumerate(data) if v == 1]
            labels.append(indices)

        with open(output_path, 'w') as f:
            for row in labels:
                for item in row:
                    f.write('{} '.format(item))
                    f.write('\n')
