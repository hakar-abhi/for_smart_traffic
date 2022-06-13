import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #prevents tensorflow from polluting standard error msgs

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')

print("No. of physical devices:", len(physical_devices))

if len(physical_devices) > 0:

    #allocating required GPU's memory for the model gradually
    tf.config.experimental.set_memory_growth(physical_devices[0],True)

from absl import app, flags, logging
from absl.flags import FLAGS

import infra.utils as utils
from infra.functions import *

from tensorflow.python.saved_model import tag_constants

from PIL import Image

import cv2 as cv

import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework','tf','platform for AI development')
flags.DEFINE_string('weights','./checkpoints/yolov3-416','can be J. Redmons YOLO weights, metadata assets or your custom model weights + metadata + assets')
flags.DEFINE_integer('size',416,'resize images to')
flags.DEFINE_string('model','yolov3','J. Redmons YOLO or your own model')
flags.DEFINE_list('images','./data/images/asanchowk.jpg','path to input image')
flags.DEFINE_string('output','./detections/','path to output folder')
flags.DEFINE_float('iou',0.45,'iou threshold')
flags.DEFINE_float('score',0.50,'score threshold')
flags.DEFINE_boolean('count',False,'count objects within images')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('dont_show',False,'do not show image output')


def main(_argv):

    config = ConfigProto()

    #allow_growth option attempts to allocate only as much GPU memory based on runtime applications: very little memory
    #at first and as sessions get run, more and more GPU memory is required , we extend the GPU memory region needed by TF process
    config.gpu_options.allow_growth = True

    session = InteractiveSession(config=config)

    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS) #FLAGS set decide stride, anchors

    input_size =FLAGS.size
    images = FLAGS.images

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])

    #loop through images in list and run YOLOV3 or your custom model on each images
    for count, image_path in enumerate(images,1):
        original_image = cv.imread(image_path)
        original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)

        image_data = cv.resize(original_image,(input_size, input_size))
        image_data = image_data/255

        image_name = image_path.split('/')[-1]
        image_name = image_name.split('.')[0]

        images_data = []
        
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        infer = saved_model_loaded.signatures['serving_default']

        batch_data = tf.constant(images_data)

        pred_bbox = infer(batch_data)

        for key,value in pred_bbox.items():

            boxes = value[:, :, 0:4]

            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0],-1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class= 50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        

        original_h, original_w, _ = original_image.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)


        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        allowed_classes = ['bicycle','car','motorbike','bus']

        if FLAGS.count:

            counted_classes = count_objects(pred_bbox, by_class=True, allowed_classes=allowed_classes)

            for key, value in counted_classes.items():
                print("Number of {}s : {}".format(key,value))
            image = utils.draw_bbox(original_image, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes)
        else:
            image = utils.draw_bbox(original_image, pred_bbox, FLAGS.info, allowed_classes=allowed_classes)
        image = Image.fromarray(image.astype(np.uint8))

        if not FLAGS.dont_show:
            image.show()
        image =cv.cvtColor(np.array(image), cv.COLOR_BGR2RGB)
        cv.imwrite(FLAGS.output + 'detection' + str(count) + '.png', image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass   
