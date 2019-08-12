import os


class ModelBase:
    INDEX_TO_STRING=dict()
    PATH_TO_MODEL=str
    PATH_TO_LABELS=str
    NUM_CLASSES=int
    LINK_TO_DOWNLOAD_MODEL=str

class MSCOCOModel(ModelBase):
    # Model 1
    INDEX_TO_STRING = index_to_string = {
                                            3: 'car',
                                            6: 'bus',
                                            8: 'truck',
                                            1: 'person',
                                            10: 'traffic light'
                                    }

    MOD_LABELS = {
                    1:  {'name': 'person', 'id':1},
                    3:  {'name': 'car', 'id': 3},
                    6:  {'name': 'bus', 'id': 6},
                    8:  {'name': 'truck', 'id': 8},
                    10:  {'name': 'traffic light', 'id': 10},
                }

    PATH_TO_MODEL =   os.path.join(os.path.dirname(__file__),
                        'models',
                        'ssdlite_mobilenet_v2_coco', #'ssdlite_mobilenet_v2_cvat_cars', 
                        'frozen_inference_graph.pb')
    
    PATH_TO_LABELS =  os.path.join(os.path.dirname(__file__),
                        'models',
                        'labels',
                        'mscoco_label_map.pbtxt') #  'car_label_map.pbtxt')   #'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90
    LINK_TO_DOWNLOAD_MODEL = "http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz"


class CVATKITTI(ModelBase):
    #'ssdlite_mobilenet_v2_cvat_cars', 
    # #  'car_label_map.pbtxt')   #'mscoco_label_map.pbtxt')
    pass

class KittiModel(ModelBase):
    # Model 2

    INDEX_TO_STRING = index_to_string = {
                                    1: 'car',
                                    2: 'pedestrian'
                                }

    MOD_LABELS = {
            1:  {'name': 'car', 'id': 1},
            2:  {'name': 'pedestrian', 'id':2}
        }

    PATH_TO_MODEL = os.path.join(os.path.dirname(__file__),
                    'models',
                    'ssdlite_mobilenet_v2_kitti',
                    'frozen_inference_graph.pb')
    
    PATH_TO_LABELS = os.path.join(os.path.dirname(__file__),
                        'models',
                        'labels',
                        'kitti_label_map.pbtxt') 

    NUM_CLASSES = 2
    LINK_TO_DOWNLOAD_MODEL = "http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz"

