import cv2
import numpy as np
import rospy
from styx_msgs.msg import TrafficLight
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

class TLClassifier(object):
    def __init__(self):
        pass

    def get_model():
        global model
        if not model:
            model = load_model('light_classifier_model.h5')
        return model

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        loaded_model = get_model()
        image_array = img_to_array(image, PIL.Image.ANTIALIAS)
        prediction = loaded_model.predict(image_array[None, :])
        if prediction[0][0] == 1:
            return TrafficLight.RED
        elif prediction[0][1] == 1:
            return TrafficLight.YELLOW
        else:
            return TrafficLight.GREEN