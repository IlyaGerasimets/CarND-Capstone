import cv2
import numpy as np
import rospy
from styx_msgs.msg import TrafficLight
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import os
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model = load_model(dir_path + '/learn/light_classifier_model.h5')
        self.graph = tf.get_default_graph()
        self.model._make_predict_function()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """        
        image = Image.fromarray(image)
        image_array = img_to_array(image.resize((64, 64), Image.ANTIALIAS))

        with self.graph.as_default():
            prediction = self.model.predict(image_array[None, :])
            
        if prediction[0][0] == 1:
            return TrafficLight.RED
        elif prediction[0][1] == 1:
            return TrafficLight.YELLOW
        else:
            return TrafficLight.GREEN