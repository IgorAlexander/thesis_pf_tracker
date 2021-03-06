# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
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
whiteLower = (0, 0, 180)
whiteUpper = (180, 26, 255)

blueLower = (115, 96, 26)
blueUpper = (130, 255, 255)

greenLower = (38, 96, 6)
greenUpper = (64, 255, 255)

yellowLower = (15, 96, 6)
yellowUpper = (35, 255, 255)

redLower1 = (0, 96, 6)
redUpper1 = (10, 255, 255)
redLower2 = (160, 96, 6)
redUpper2 = (179, 255, 255)
track_pts = {}

first_frame = True
particles_matrix = []
tracks_arr = []

writer = None
fourcc = cv2.VideoWriter_fourcc(*"MJPG")

n_frame = 1

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
	#mask1 = cv2.inRange(hsv, redLower1, redUpper1)
	#mask2 = cv2.inRange(hsv, redLower2, redUpper2)

	#mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0.0)

	mask1 = cv2.inRange(hsv, whiteLower, whiteUpper)
	mask2 = cv2.inRange(hsv, blueLower, blueUpper)

	# mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0.0)

	# ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY_INV)
	# mask = cv2.erode(mask, None, iterations=1)
	mask1 = cv2.dilate(mask1, None, iterations=6)
	mask1 = cv2.erode(mask1, None, iterations=2)

	mask2 = cv2.dilate(mask2, None, iterations=6)
	mask2 = cv2.erode(mask2, None, iterations=2)

	#DETECTION

	cnts = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	pparticles_arr = []

	# First TEAM
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		player_count = 0
		for cnt in cnts:

			x,y,w,h = cv2.boundingRect(cnt)

			# only proceed if the radius meets a minimum size
			if h > 12 and h < 100 and w > 5 and w < 40 and y > 30:

				M = cv2.moments(cnt)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				# draw the circle and centroid on the frame,
				# then update the list of tracked points

				# Init tracks
				if first_frame:
					track_tmp = pf.Track(center[1], center[0], team = 1)
					track_tmp.assignRefHistogram(frame, x, y, w, h, player_count)
					tracks_arr.append(track_tmp)
					player_count = player_count + 1
					track_pts[track_tmp] = deque(maxlen=pf._BUFFER_TRACK)
				else:
					inner_track = pf.findInnerTrack(tracks_arr, x, y, w, h, 1)
					if inner_track != None:
						inner_track.update_center(center[0], center[1])
						cv2.putText(frame, str(inner_track.number), (x + 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
						if n_frame % pf._R_UPDATE_RATE == 0:
							inner_track.assignRefHistogram(frame, x, y, w, h, player_count)						

				pparticles_arr.extend(pf.findInnerParticles(particles_matrix, x, y, w, h))

				cv2.rectangle(frame, (int(x), int(y)), (int(x)+int(w), int(y)+int(h)),
					(255, 255, 255), 2)
				cv2.circle(frame, center, 2, (0, 0, 255), -1)

	if len(cnts2) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		for cnt in cnts2:

			x,y,w,h = cv2.boundingRect(cnt)

			# only proceed if the radius meets a minimum size
			if h > 12 and h < 100 and w > 5 and w < 40 and y > 30:

				M = cv2.moments(cnt)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				# draw the circle and centroid on the frame,
				# then update the list of tracked points

				# Init tracks
				if first_frame:
					track_tmp = pf.Track(center[1], center[0], team = 2)
					track_tmp.assignRefHistogram(frame, x, y, w, h, player_count)
					tracks_arr.append(track_tmp)
					player_count = player_count + 1
					track_pts[track_tmp] = deque(maxlen=pf._BUFFER_TRACK)
				else:
					inner_track = pf.findInnerTrack(tracks_arr, x, y, w, h, 2)
					if inner_track != None:
						inner_track.update_center(center[0], center[1])
						cv2.putText(frame, str(inner_track.number), (x + 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
						if n_frame % pf._R_UPDATE_RATE == 0:
							inner_track.assignRefHistogram(frame, x, y, w, h, player_count)						

				pparticles_arr.extend(pf.findInnerParticles(particles_matrix, x, y, w, h))

				cv2.rectangle(frame, (int(x), int(y)), (int(x)+int(w), int(y)+int(h)),
					(255, 0, 0), 2)
				cv2.circle(frame, center, 2, (0, 0, 255), -1)

		first_frame = False

	pparticles_set = set(pparticles_arr)
	pf.draw_tracks(frame, tracks_arr)

	# Iteration of the particle filter model

	g = {}
	f = {}
	p = {}

	for s_temp in pparticles_set:
		g[s_temp] = []

	for x_temp in tracks_arr:
		# Initialize f
		f[x_temp] = []

		# Line tracks
		track_pts[x_temp].appendleft(x_temp.p)

		for i in xrange(1, len(track_pts[x_temp])):
			# if either of the tracked points are None, ignore
			# them
			if track_pts[x_temp][i - 1] is None or track_pts[x_temp][i] is None:
				continue
			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(pf._BUFFER_TRACK / float(i + 1)) * 1)
			cv2.line(frame, track_pts[x_temp][i - 1], track_pts[x_temp][i], (0, 0, 255), thickness)

	for x_temp in tracks_arr:

		x_temp.predict();

		# Kalman filter prediction
		for s_temp in pparticles_set:
			if s_temp.isNearby(x_temp):
				# Likelihood model
				p_color = pf.calcColorProb(frame, s_temp, x_temp)				
				p_motion = pf.calcMotionProb(frame, s_temp, x_temp)
				p[(s_temp, x_temp)] = p_color * p_motion


				f[x_temp].append(s_temp)
				g[s_temp].append(x_temp)

	for x_temp in tracks_arr:
		if len(f[x_temp]) == 0:
			nearby_particles = x_temp.getNearbyParticles(particles_matrix)
			pparticles_set.update(nearby_particles)

			for s_temp in nearby_particles:
				p_color = pf.calcColorProb(frame, s_temp, x_temp)
				p_motion = pf.calcMotionProb(frame, s_temp, x_temp)
				p[(s_temp, x_temp)] = p_color * p_motion

				if s_temp not in g:
					g[s_temp] = []

				f[x_temp].append(s_temp)
				g[s_temp].append(x_temp)


	for s_temp in pparticles_set:
		if len(g[s_temp]) > 0:
			w_s = {}
			for x_temp in g[s_temp]:
				w_s[x_temp] = p[(s_temp, x_temp)]
			w_s_max = max(w_s.values())
			for x_temp in g[s_temp]:
				if w_s[x_temp] == w_s_max:
					p[(s_temp, x_temp)] = p[(s_temp, x_temp)] * w_s[x_temp]
				else:
					f[x_temp].remove(s_temp)

	for x_temp in tracks_arr:
		if len(f[x_temp]) > 0:
			weights = {}
			for s_temp in f[x_temp]:
				weights[s_temp] = p[(s_temp, x_temp)]
			
			weights = pf.dict_normalize(weights)
			p_obs = (0,0)
			for s_temp in f[x_temp]:
				p_obs = tuple([(p_obs[i] + weights[s_temp]*s_temp.q[i]) for i in range(2)])

			x_noise = np.random.normal(0, pf._SHOULDER_WIDTH / 2.0)
			y_noise = np.random.normal(0, pf._SHOULDER_WIDTH)

			old_p = (x_temp.p[0], x_temp.p[1])			
			
			x_temp.p = (int(round(p_obs[0] + x_noise)), int(round(p_obs[1] + y_noise)))

			#print str(old_p) + "vs" + str(x_temp.p)

	# pf.draw_pos_particles(frame, pparticles_set)
	# pf.draw_particles(frame, particles_matrix)
	#	pf.draw_tracks(frame, tracks_arr)
	# show the frame to our screen
	cv2.imshow("frame", frame)
	cv2.imshow("mask1", mask1)
	cv2.imshow("mask2", mask2)

	if writer is None:
		(h, w) = frame.shape[:2]
		writer = cv2.VideoWriter("output.avi", fourcc, 20, (w, h), True)

	writer.write(frame)

	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

	n_frame = n_frame + 1

# cleanup the camera and close any open windows
camera.release()
writer.release()
cv2.destroyAllWindows()