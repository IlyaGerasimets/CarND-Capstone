import os
import numpy as np
import tensorflow as tf
import label_map_util
import rospy
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        pwd = os.path.dirname(os.path.realpath(__file__))
        # pretrained on bosch dataset with 14 classes
        label_map = label_map_util.load_labelmap(pwd + "/label_map.pbtxt")
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=13, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        # load model
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(pwd + "/frozen_inference_graph.pb", 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # tensors for graph
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """        
        image_array = np.expand_dims(image, axis=0)

        with self.graph.as_default():
             with tf.Session(graph=self.graph) as sess:
                (scores, classes) = sess.run([self.detection_scores, self.detection_classes], feed_dict={self.image_tensor: image_array})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        if len(scores) > 0:
            max_idx = scores.argmax(axis=0)
            class_name = self.category_index[classes[max_idx]]['name']
            if class_name == 'Red':
                return TrafficLight.RED
            elif class_name == 'Green':
                return TrafficLight.GREEN
            elif class_name == 'Yellow':
                return TrafficLight.YELLOW
            else:
                return TrafficLight.UNKNOWN
        else:
            return TrafficLight.UNKNOWN