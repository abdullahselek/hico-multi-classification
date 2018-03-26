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

    def __prepare_output_texts(self,
                               label_text_file,
                               list_action):
        """Prepare files for output texts.
        Args:
          label_text_file (str):
            File path to create output texts.
          list_action (struct):
            Each entry is one HOI category
        """

        with open(label_text_file, 'w') as f:
		        for i, row in enumerate(list_action):
			          obj = row['nname'][0]
			          obj = ''.join(map(str, obj))
			          verb = row['vname'][0]
			          verb = ''.join(map(str, verb))
			          f.write('{} {} {}\n'.format(i, obj, verb))

    def __generate_filenames_file(self, filenames_split, output_path):
        """Generate files for train and test files.
        Args:
          filenames_split (array):
            Entry is a file name of an training or test image.
          output_path (str):
            File path for output file.
        """

        with open(output_path, 'w') as f:
		        for row in filenames_split:
			          filename = map(str, row[0])
			          filename = ''.join(filename)
			          f.write('{}\n'.format(filename))

    def process(self):
        """Start processing train and test files."""

        # Read the labels from annotation mat file
        anno_mat = self.__load_mat(self._data_dir, self._anno_mat)
        # Output labels
        anno_train = anno_mat['anno_train']
        labels_train_file = os.path.join(self._data_dir, 'labels_train.txt')
        self.__generate_labels_file(anno_train, labels_train_file)
        anno_test = anno_mat['anno_test']
        labels_test_file = os.path.join(self._data_dir, 'labels_test.txt')
        self.__generate_labels_file(anno_test, labels_test_file)
        # Output label_text
        list_action = anno_mat['list_action']
        label_text_file = os.path.join(self._data_dir, 'label_text.txt')
        self.__prepare_output_texts(label_text_file, list_action)
        # Output filenames
        filenames_train = anno_mat['list_train']
        filenames_train_file = os.path.join(self._data_dir, 'filenames_train.txt')
        self.__generate_filenames_file(filenames_train, filenames_train_file)
        filenames_test = anno_mat['list_test']
        filenames_test_file = os.path.join(self._data_dir, 'filenames_test.txt')
        self.__generate_filenames_file(filenames_test, filenames_test_file)
