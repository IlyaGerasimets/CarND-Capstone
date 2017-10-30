import cv2
import numpy as np
import rospy
from styx_msgs.msg import TrafficLight
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import os
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model = load_model(dir_path + '/learn/nets/light_classifier_model.h5')
        # self.graph = tf.get_default_graph()
        self.model._make_predict_function()

    def get_classification(self, cv_image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """        
        cv_image = cv2.resize(cv_image, (160, 120)).astype(np.float32)
        image_data = np.reshape(cv_image, (1,160,120,3))
        x = preprocess_input(image_data)
        prediction = self.model.predict(x)
            
        rospy.loginfo(prediction[0])

        threshold = 0.95

        if prediction[0][0] > threshold:
            return TrafficLight.GREEN
        elif prediction[0][1] > threshold:
            return TrafficLight.YELLOW
        elif prediction[0][2] > threshold:
            return TrafficLight.RED
        else:
            return TrafficLight.UNKNOWN