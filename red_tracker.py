# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import particle_filter as pf

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (38, 96, 6)
greenUpper = (64, 255, 255)

yellowLower = (15, 96, 6)
yellowUpper = (35, 255, 255)

redLower1 = (0, 96, 6)
redUpper1 = (10, 255, 255)
redLower2 = (160, 96, 6)
redUpper2 = (179, 255, 255)
pts = deque(maxlen=args["buffer"])

first_frame = True
particles_matrix = []
tracks_arr = []

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	if first_frame:
		particles_matrix = pf.init_particles(frame)

	# resize the frame, blur it, and convert it to the HSV
	# color space
	#frame = imutils.resize(frame, width=600)
	#blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask1 = cv2.inRange(hsv, redLower1, redUpper1)
	mask2 = cv2.inRange(hsv, redLower2, redUpper2)

	mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0.0)

	# mask3 = cv2.inRange(hsv, yellowLower, yellowUpper)
	# 
	# mask = cv2.addWeighted(mask, 1.0, mask3, 1.0, 0.0)

	# ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY_INV)
	mask = cv2.erode(mask, None, iterations=1)
	mask = cv2.dilate(mask, None, iterations=1)

	#DETECTION

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	pparticles_arr = []


	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		player_count = 0
		for cnt in cnts:

			x,y,w,h = cv2.boundingRect(cnt)

			# only proceed if the radius meets a minimum size
			if h > 1 and h < 30 and w > 1 and w < 20 and y > 40:

				M = cv2.moments(cnt)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				# draw the circle and centroid on the frame,
				# then update the list of tracked points

				# Init tracks
				if first_frame:
					track_tmp = pf.Track(center[1], center[0])
					track_tmp.assignRefHistogram(frame, x, y, w, h, player_count)
					tracks_arr.append(track_tmp)
					player_count = player_count + 1

				pparticles_arr.extend(pf.findInnerParticles(particles_matrix, x, y, w, h))

				cv2.rectangle(frame, (int(x), int(y)), (int(x)+int(w), int(y)+int(h)),
					(0, 255, 255), 2)
				cv2.circle(frame, center, 2, (0, 0, 255), -1)

		first_frame = False

	pparticles_set = set(pparticles_arr)

	# Iteration of the particle filter model

	g = {}
	f = {}
	p = {}

	for x_temp in tracks_arr:
		# Kalman filter prediction
		for s_temp in pparticles_set:
			if s_temp.isNearby(x_temp):
				# Likelihood model
				p_color = pf.calcColorProb(frame, s_temp, x_temp)
				p_motion = pf.calcMotionProb(frame, s_temp, x_temp)
				p[(s_temp, x_temp)] = p_color * p_motion

				f[x_temp] = s_temp
				g[s_temp] = x_temp

	pf.draw_pos_particles(frame, pparticles_set)
	# pf.draw_particles(frame, particles_matrix)
	# pf.draw_tracks(frame, tracks_arr)
	# show the frame to our screen
	cv2.imshow("frame", frame)
	cv2.imshow("mask", mask)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()