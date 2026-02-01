
import numpy as np
import os
from xml.etree import ElementTree
from .auxiliary import Auxiliary

class XML_preprocessor(object):
    """
    This class read Annotation and prepares
    """
    def __init__(self, data_path, num_classes, normalize=False):
        self.auxiliary = Auxiliary()
        self.path_prefix = data_path
        self.num_classes = num_classes
        self.normalization = normalize
        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        path = rf"{self.path_prefix}"
        filename = os.path.basename(path)
        tree = ElementTree.parse(path)
        root = tree.getroot()
        bounding_boxes = []
        one_hot_classes = []
        size_tree = root.find('size')
        width = float(size_tree.find('width').text)
        height = float(size_tree.find('height').text)
        print(f"width: {width}, height: {height}")
        for object_tree in root.findall('object'):
            for bounding_box in object_tree.iter('bndbox'):
                if self.normalization:
                    xmin = float(bounding_box.find('xmin').text) / width
                    ymin = float(bounding_box.find('ymin').text) / height
                    xmax = float(bounding_box.find('xmax').text) / width
                    ymax = float(bounding_box.find('ymax').text) / height
                else:
                    xmin = float(bounding_box.find('xmin').text)
                    ymin = float(bounding_box.find('ymin').text)
                    xmax = float(bounding_box.find('xmax').text)
                    ymax = float(bounding_box.find('ymax').text)
            bounding_box = [xmin, ymin, xmax, ymax]
            bounding_boxes.append(bounding_box)
            class_name = object_tree.find('name').text
            one_hot_class = self._to_one_hot(class_name)
            one_hot_classes.append(one_hot_class)
            # image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[filename] = image_data

    def _to_one_hot(self, name):
        one_hot_vector = [0] * self.num_classes
        if name == 'A':
            one_hot_vector[0] = 1
        elif name == 'B':
            one_hot_vector[1] = 1
        elif name == 'E':
            one_hot_vector[2] = 1
        elif name == 'G':
            one_hot_vector[3] = 1
        else:
            msg = f"Unknown label: {name}"
            self.auxiliary.log(msg)

        return one_hot_vector

## example on how to use it
# import pickle
# Data = XML_preprocessor('VOC2007/Annotations/').Data
# pickle.dump(Data,open('VOC2007.p','wb'))
