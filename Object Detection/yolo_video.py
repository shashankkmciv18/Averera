# USAGE
# py yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco
# py yolo_video.py --output output/.avi --yolo yolo-coco
# py yolo_video.py --yolo yolo-coco
# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.6,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# takes yolo model which is already a pretrained model. along with it, coco names are names of those 80 objects
# initialize a list of colors to represent each possible class label
np.random.seed(42)
#random colors are taken
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# Darknet has the predefined weights
ln = net.getLayerNames()
# get last layer details bcoz it is the last layer which detects classes
# print(ln)
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#print(ln)

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
# imutils is function for basic image processing
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	# counts frames
	total = int(vs.get(prop)) #get returns prop value
	print("[INFO] {} total frames in video".format(total))
# all this counts frames to calculate elapsed time
# an error occurred while trying to determine the total
# number of frames in the video file

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1



# import the necessary packages
#from imutils import paths

#def find_marker(frame):
	# convert the image to grayscale, blur it, and detect edges
#	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#	gray = cv2.GaussianBlur(gray, (5, 5), 0)
#	edged = cv2.Canny(gray, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
#	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#	cnts = imutils.grab_contours(cnts)
#	c = max(cnts, key = cv2.contourArea)
	# compute the bounding box of the of the paper region and return it
#	return cv2.minAreaRect(c)

#def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
#	return (knownWidth * focalLength) / perWidth




# loop over frames from the video file stream
while True:
	# read the next frame from the file
# grabbed shows whether it is taking video or not. if yes grabbed = true
	(grabbed, frame) = vs.read()
	screen_res = 600, 400
	scale_width = screen_res[0] / frame.shape[1]
	scale_height = screen_res[1] / frame.shape[0]
	scale = min(scale_width, scale_height)
	window_width = int(frame.shape[1] * scale)
	window_height = int(frame.shape[0] * scale)
	cv2.namedWindow('raw', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('raw', window_width, window_height)
	cv2.imshow('raw', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
# blob stands for binary large object
	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

#				print('{} - %.2f'.format(classIDs, confidences))

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
			print(text)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

		#(grabbed1, frame1) = writer.read()

		#writer = cv2.VideoWriter(args["output"], fourcc, 30,
		#						 (frame.shape[1], frame.shape[0]), True)
		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)
	window_width1 = int(frame.shape[1] * scale)
	window_height1 = int(frame.shape[0] * scale)
	cv2.namedWindow('frameresize1', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frameresize1', window_width1, window_height1)
	cv2.imshow('frameresize1', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



	# initialize the known distance from the camera to the object, which
	# in this case is 24 inches
#	KNOWN_DISTANCE = 24.0
#	# initialize the known object width, which in this case, the piece of
	# paper is 12 inches wide
#	KNOWN_WIDTH = 11.0
	# load the furst image that contains an object that is KNOWN TO BE 2 feet
	# from our camera, then find the paper marker in the image, and initialize
	# the focal length
#	marker = find_marker(frame)
#	focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

	# loop over the images
#	for imagePath in sorted(paths.list_images("images")):
		# load the image, find the marker in the image, then compute the
		# distance to the marker from the camera
#		frame = cv2.imread(imagePath)
#		marker = find_marker(frame)
#		inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
		# draw a bounding box around the image and display it
#		box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
#		box = np.int0(box)
#		cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
#		cv2.putText(frame, "%.2fft" % (inches / 12),
#					(frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
#					2.0, (0, 255, 0), 3)
#		cv2.imshow("image", frame)
#		cv2.waitKey(0)



# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
cv2.destroyAllWindows()
