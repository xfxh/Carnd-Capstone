from styx_msgs.msg import TrafficLight
import tensorflow as tf
import rospy
import numpy as np
import os
import cv2

MAX_IMAGE_WIDTH = 300
MAX_IMAGE_HEIGHT = 300

class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier
        self.current_light = TrafficLight.UNKNOWN
        self.is_site = is_site
        self.category_index = {1: TrafficLight.RED,
                               2: TrafficLight.YELLOW,
                               3: TrafficLight.GREEN,
                               4: TrafficLight.UNKNOWN}

        self.min_score_thresh = 0.5
        #find the correct model path
        cwd = os.path.dirname(os.path.realpath(__file__))
        if self.is_site:
            model_filename = cwd + '/model/real_frozen_inference_graph.pb'
        else:
            model_filename = cwd + '/model/sim_frozen_inference_graph.pb'

        #start to load frozen model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_filename, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        #create tf session for detection
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.detection_graph, config=config)



        #define input and output Tensor for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        #each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        #each score represents how level of confidence for each of the objects.
        #score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        #process image
        image = cv2.resize(image, (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims(image_rgb, axis = 0)

        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes,
                                                          self.num_detections], feed_dict ={self.image_tensor: image_np})
        
        
        boxes_sque = np.squeeze(boxes)
        scores_sque = np.squeeze(scores)
        classes_sque = np.squeeze(classes)
        num_sque = np.squeeze(num)

        count_red = 0
        count_green = 0
        
        for i in range(num_sque):
            if scores_sque[i] > self.min_score_thresh:
                class_name = self.category_index[classes_sque[i]]

                if class_name == TrafficLight.RED:
                    rospy.logdebug("RED score:%f", scores_sque[i])
                elif class_name == TrafficLight.GREEN:
                    rospy.logdebug("GREEN score:%f", scores_sque[i])
                elif class_name == TrafficLight.YELLOW:
                    rospy.logdebug("YELLOW score:%f", scores_sque[i])
                else:
                    rospy.logdebug("UNKNOWN score:%f", scores_sque[i])
                self.current_light = class_name
                return self.current_light

        return TrafficLight.UNKNOWN
