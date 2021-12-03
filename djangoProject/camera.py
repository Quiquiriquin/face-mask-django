import cv2
import threading
import imutils
import numpy as np
import os
import pickle
import tensorflow as tf
from django.conf import settings
from imutils.video import FPS
from imutils.video import VideoStream
from scipy.special import softmax
from djangoProject import extract_embeddings

# load our serialized face detector model from disk
protoPath = os.path.join(settings.BASE_DIR, "models/deploy.prototxt.txt")
modelPath = os.path.join(settings.BASE_DIR, "models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# load our serialized face embedding model from disk
model = tf.keras.models.load_model(os.path.join(settings.BASE_DIR,'models/face_cnn_model'))
# embedder = cv2.dnn.readNetFromTorch(os.path.join(settings.BASE_DIR,'models/face_cnn_model'))
# load the actual face recognition model along with the label encoder
# recognizer = os.path.sep.join([settings.BASE_DIR, "output\\recognizer.pickle"])
# recognizer = pickle.loads(open(recognizer, "rb").read())
# le = os.path.sep.join([settings.BASE_DIR, "output\\le.pickle"])
# le = pickle.loads(open(le, "rb").read())
# dataset = os.path.sep.join([settings.BASE_DIR, "dataset"])
# user_list = [ f.name for f in os.scandir(dataset) if f.is_dir() ]

labels = ['Mask', 'No Mask', 'Uncorrect', 'Uncorrect']


def getColor(label):
	if label == "Mask":
		color = (0, 255, 0)

	else:
		color = (0, 0, 255)

	return color


class FaceDetect(object):
	def __init__(self):
		# initialize the video stream, then allow the camera sensor to warm up
		self.vs = cv2.VideoCapture(0)
		# start the FPS throughput estimator
		# self.fps = self.vs.read()
		(self.grabbed, self.frame) = self.vs.read()
		threading.Thread(target=self.update, args=()).start()


	def __del__(self):
		self.vs.release()


	def update(self):
		while True:
			(self.grabbed, self.frame) = self.vs.read()


	def get_frame(self):
		# grab the frame from the threaded video stream
		image = self.frame
		# resize the frame to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		(h, w) = image.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				pt1 = (startX, startY)
				pt2 = (endX, endY)
				# extract the face ROI
				face = image[startY:endY, startX:endX]
				if len(face) == 0:
					return

				face_blob = cv2.dnn.blobFromImage(face, 1, (100, 100), (104, 117, 123), swapRB=True)
				face_blob_squeeze = np.squeeze(face_blob).T
				face_blob_rotate = cv2.rotate(face_blob_squeeze, cv2.ROTATE_90_CLOCKWISE)
				face_blob_flip = cv2.flip(face_blob_rotate, 1)
				img_norm = np.maximum(face_blob_flip, 0) / face_blob_flip.max()
				img_input = img_norm.reshape(1, 100, 100, 3)
				result = model.predict(img_input)
				result = softmax(result)[0]
				confidence_index = result.argmax()
				confidence_score = result[confidence_index]
				label = labels[confidence_index]
				label_text = '{} : {:,.0f} %'.format(label, confidence_score * 100)
				color = getColor(label)
				cv2.rectangle(image, pt1, pt2, color, 1)
				cv2.putText(image, label_text, pt1, cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
		# self.fps.update()
		ret, jpeg = cv2.imencode('.jpg', image)
		return jpeg.tobytes()
