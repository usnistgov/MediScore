# -*- coding: utf-8 -*-
import sys
import pickle
from collections import OrderedDict

class DataContainer:
    data_container_version = "2.0"

	def __init__(self, fa_array, fn_array, threshold, label=None, line_options=None, fa_label=None, fn_label=None):
		self.fa = fa_array # False Alarm, equivalent for False Positive
		self.fn = fn_array # False Negative, equivalent for Miss
        self.threshold = threshold
        self.label = label
        self.fa_label = fa_label
        self.fn_label = fn_label

        if line_options is not None:
            self.line_options = line_options
        else:
            self.line_options = DataContainer.get_default_line_options()
            self.line_options["label"] = label

	def set_default_line_options(self):
		self.line_options = DataContainer.get_default_line_options()

	def dump(self, file_name):
        """Serialize the object (formatted in a binary)
        file_name: Dump file name
        """        
        file = open(file_name, 'wb')
        pickle.dump(self, file)
        file.close()

    @staticmethod
    def get_default_line_options():
        """ Creation of defaults line options dictionnary
        """
        return OrderedDict([('color', 'red'),
                            ('linestyle', 'solid'),
                            ('marker', '.'),
                            ('markersize', 5),
                            ('markerfacecolor', 'red'),
                            ('label', None),
                            ('antialiased', 'False')])

    @staticmethod
    def load_file(path):
    """ Load Dumped files
        path: absolute path to the file
    """
    with open(path, 'rb') as file:
        if sys.version_info[0] >= 3:
            obj = pickle.load(file, encoding='latin1') 
        else:
            obj = pickle.load(file)
    return obj

    @staticmethod
    def aggregate(dc_list, output_label="Average", method="average", average_resolution=500, line_options=None):

        if line_options is None:
            default_line_options = DataContainer.get_default_line_options()
            default_line_options["color"] = "green"

        if method == "average":
            x = np.linspace(0, 1, average_resolution)
            ys = [np.interp(x, data.fa, data.fn) for data in dc_list]
            return DataContainer(x, np.vstack(ys).mean(0), label="output_label", line_options=line_options)
