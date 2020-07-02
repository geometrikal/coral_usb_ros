#!/usr/bin/env python


import matplotlib

matplotlib.use("Agg")  # NOQA
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys

# OpenCV import for python3.5
sys.path.remove('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA
import cv2  # NOQA

sys.path.append('/opt/ros/{}/lib/python2.7/dist-packages'.format(os.getenv('ROS_DISTRO')))  # NOQA

# from chainercv.visualizations import vis_bbox
from cv_bridge import CvBridge
from edgetpu.classification.engine import ClassificationEngine
import PIL.Image
import rospkg
import rospy

from dynamic_reconfigure.server import Server
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import Rect
from jsk_recognition_msgs.msg import RectArray
from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import Image

from coral_usb.cfg import EdgeTPUPatchDetectorConfig


class EdgeTPUPatchDetector(ConnectionBasedTransport):

    def __init__(self):
        super(EdgeTPUPatchDetector, self).__init__()
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('coral_usb')
        self.bridge = CvBridge()
        self.classifier_name = rospy.get_param(
            '~classifier_name', rospy.get_name())
        model_file = os.path.join(pkg_path, './models/patch.tflite')
        label_file = os.path.join(pkg_path, './models/patch_labels.txt')
        model_file = rospy.get_param('~model_file', model_file)
        label_file = rospy.get_param('~patch_label_file', label_file)

        self.engine = ClassificationEngine(model_file)
        self.label_ids, self.label_names = self._load_labels(label_file)

        # dynamic reconfigure
        self.srv = Server(EdgeTPUPatchDetector, self.config_callback)

        # Published task
        self.pub_classification = self.advertise('~output/class', ClassificationResult, queue_size=1)

    def subscribe(self):
        self.sub_image = rospy.Subscriber('~input', Image, self.image_cb, queue_size=1, buff_size=2 ** 26)

    def unsubscribe(self):
        self.sub_image.unregister()

    @property
    def visualize(self):
        return self.pub_image.get_num_connections() > 0

    def config_callback(self, config, level):
        self.score_thresh = config.score_thresh
        self.patch_height = config.patch_height
        self.patch_width = config.patch_width
        self.subsample = config.subsample
        return config

    def _load_labels(self, path):
        p = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())
            labels = {int(num): text.strip() for num, text in lines}
            return list(labels.keys()), list(labels.values())

    def image_cb(self, msg):
        img = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'))
        H, W = img.shape[:2]
        h_step = self.patch_height * self.subsample
        w_step = self.patch_width * self.subsample
        h_idxs = np.arange(0, H, h_step)
        w_idxs = np.arange(0, W, h_step)

        results = []
        for y in h_idxs:
            for x in w_idxs:
                sub_img = img[y:y+h_step:self.subsample, x:x+w_step:self.subsample, :]
                cls = self.engine.classify_with_input_tensor(sub_img, threshold=0.0, top_k=1)
                results.append(cls)

        labels = [p[0][0] for p in results]
        probs = [p[0][1] for p in results]

        cls_msg = ClassificationResult(
            header=msg.header,
            classifier=self.classifier_name,
            target_names=self.label_names,
            labels=labels,
            label_names=[self.label_names[l] for l in labels],
            label_proba=probs)

        self.pub_classification.publish(cls_msg)

        # if self.visualize:
        #     fig = plt.figure(
        #         tight_layout={'pad': 0})
        #     ax = plt.Axes(fig, [0., 0., 1., 1.])
        #     ax.axis('off')
        #     fig.add_axes(ax)
        #     vis_bbox(
        #         img.transpose((2, 0, 1)),
        #         bboxes, labels, scores,
        #         label_names=self.label_names, ax=ax)
        #     fig.canvas.draw()
        #     w, h = fig.canvas.get_width_height()
        #     vis_img = np.fromstring(
        #         fig.canvas.tostring_rgb(), dtype=np.uint8)
        #     vis_img.shape = (h, w, 3)
        #     fig.clf()
        #     plt.close()
        #     vis_msg = self.bridge.cv2_to_imgmsg(vis_img, 'rgb8')
        #     # BUG: https://answers.ros.org/question/316362/sensor_msgsimage-generates-float-instead-of-int-with-python3/  # NOQA
        #     vis_msg.step = int(vis_msg.step)
        #     vis_msg.header = msg.header
        #     self.pub_image.publish(vis_msg)


if __name__ == '__main__':
    rospy.init_node('edgetpu_patch_detector')
    detector = EdgeTPUPatchDetector()
    rospy.spin()
