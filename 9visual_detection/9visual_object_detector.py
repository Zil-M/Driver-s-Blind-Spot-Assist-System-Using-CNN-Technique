import cv2
import os
import os.path as path
import glob
import imageio
import numpy as np
import tensorflow as tf
from keras.models import load_model
import category
import aserial
import threading
import tarfile
import six.moves.urllib as urllib
import time

from objet_detection.utils import label_map_util
from objet_detection.utils import visualization_utils as vis_util
from objet_detection.utils import ops as utils_ops


global Lider_value
global Sonar_L_value
global Sonar_R_value

class ObjectDetector():
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    GRAPH_FILE_NAME = 'frozen_inference_graph.pb'
    NUM_CLASSES = 90

    def download_model(self, model_name):
        model_file = model_name + '.tar.gz'
        print("downloading model", model_name, "...")
        opener = urllib.request.URLopener()
        opener.retrieve(self.DOWNLOAD_BASE + model_file, model_file)
        print("download completed");
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if self.GRAPH_FILE_NAME in file_name:
                tar_file.extract(file, os.getcwd())
                print(self.graph_file, "is extracted");

    def __init__(self, model_name, label_file='data/mscoco_label_map.pbtxt'):
        print("ObjectDetector('%s', '%s')" % (model_name, label_file))
        self.process_this_frame = True

        self.graph_file = model_name + '/' + self.GRAPH_FILE_NAME
        if not os.path.isfile(self.graph_file):
            self.download_model(model_name)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(self.graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            graph = self.detection_graph

            ops = graph.get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
              ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = graph.get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, 480, 640)
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            self.tensor_dict = tensor_dict

        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        label_map = label_map_util.load_labelmap(label_file)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.output_dict = None

        self.last_inference_time = 0

    def run_inference(self, image_np):
        sess = self.sess
        graph = self.detection_graph
        with graph.as_default():
            image_tensor = graph.get_tensor_by_name('image_tensor:0')

            output_dict = sess.run(self.tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image_np, 0)})

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict

    def time_to_run_inference(self):
        unixtime = int(time.time())
        if self.last_inference_time != unixtime:
            self.last_inference_time = unixtime
            return True
        return False

    def detect_objects(self, frame):
        time1 = time.time()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        time2 = time.time()

        if self.time_to_run_inference():
            self.output_dict = self.run_inference(rgb_small_frame)

        time3 = time.time()

        vis_util.visualize_boxes_and_labels_on_image_array(
          frame,
          self.output_dict['detection_boxes'],
          self.output_dict['detection_classes'],
          self.output_dict['detection_scores'],
          self.category_index,
          instance_masks=self.output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=1)

        time4 = time.time()
        dic_list = self.category_index[self.output_dict['detection_classes'][0]]
        object_name = dic_list['name']

        frame_b = frame.copy()
        frame_r = category.Lider_system(Lider_value, object_name, frame_b)
        category.Sonar_system(Sonar_L_value, Sonar_R_value, frame_r)			   

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()


def thread_run():
    Lider_value = aserial.GetSensorData('Lidar', 'value')
    Sonar_L_value = aserial.GetSensorData('Sonar_L', 'value')
    Sonar_R_value = aserial.GetSensorData('Sonar_R', 'value')
    print("Load_Sensor_Value")
    threading.Timer(1.0, thread_run).start()
thread_run()

if __name__ == '__main__':
    import camera

    
    detector = ObjectDetector('mask_rcnn_inception_v2_coco_2018_01_28')
    model_v = tf.keras.models.load_model('../9Visual_model.h5')
    

    cam = cv2.VideoCapture(0)
    cam1 = cv2.VideoCapture(1)

    print("program start!! press `q` to quit")
    while True:
        ret, frame = cam.read()
        ret1, frame1 = cam1.read()                    
          
        frame = detector.detect_objects(frame)

        image_v = cv2.resize(frame1, dsize=(128,128))
        cv2.imwrite('../image_storage/image1.jpg', image_v)
    
        image_path = "../image_storage/"
        file_path = glob.glob(path.join(image_path,'*.jpg'))
    
        image_listimage_list = [imageio.imread(path) for path in file_path]
        image_list = np.asarray(image_listimage_list)
        image_list = image_list/255
    
        result = model_v.predict_classes(image_list)
        str = "9 Visual type : %d" % result
        print(str)
        cv2.putText(frame1, str, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        cv2.imshow("9 visual", frame1)
            
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        if key == ord("q"):
            break
                 
    cv2.destroyAllWindows()
    print('finish')
