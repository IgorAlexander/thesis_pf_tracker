from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import math
import cv2

# Particle's box size
_BOX_WIDTH = 3
_BOX_HEIGHT = 6

# numbers of bins per channel
_CH = 10
_CS = 5
_CV = 5

# max distance between a track and particle
_R_MAX = 12

_SHOULDER_WIDTH = 1

class Particle:

	def __init__(self, weight, y, x, hist_sup = None, hist_inf = None):
		self.weight = weight
		self.q = (x, y)
		# self.scale = scale
		self.hist_sup = hist_sup
		self.hist_inf = hist_inf

	def obtainHistogram(self, frame):
		x = self.q[0]
		y = self.q[1]

		# Lets divide the box in two regions

		roi_sup = frame[y-_BOX_HEIGHT:y-_BOX_HEIGHT//2, x:x+_BOX_WIDTH]
		roi_inf = frame[y-_BOX_HEIGHT//2:y, x:x+_BOX_WIDTH]
		ref_hist_sup = cv2.calcHist([roi_sup], [0, 1, 2], None, [_CH, _CS, _CV], [0, 180, 0, 256, 0, 256])
		ref_hist_inf = cv2.calcHist([roi_inf], [0, 1, 2], None, [_CH, _CS, _CV], [0, 180, 0, 256, 0, 256])
		
		self.hist_sup = cv2.normalize(ref_hist_sup, self.hist_sup)
		self.hist_sup = self.hist_sup.flatten()

		self.hist_inf = cv2.normalize(ref_hist_inf, self.hist_inf)
		self.hist_inf = self.hist_inf.flatten()

	def isNearby(self, track):
		return (dist.euclidean(track.p, self.q) < _R_MAX)

class Track:

	def __init__(self, y, x, v = 0, R = None):
		self.p = (x, y)
		self.v = v
		self.R = R

	def assignRefHistogram(self, frame, x, y, w, h, i):
		# Extract region of interest
		roi = frame[y:y+h, x:x+w]
		
		#Show players extracted
		#cv2.imshow("aroi" + str(i), roi)
		
		# Extract HSV histogram of the roi
		ref_hist = cv2.calcHist([roi], [0, 1, 2], None, [_CH, _CS, _CV], [0, 180, 0, 256, 0, 256])
		
		# Normalize for further operations
		self.R = cv2.normalize(ref_hist, self.R)
		self.R = self.R.flatten()

		# Testing purpose
		# print "3D histogram shape: %s, with %d values" % (ref_hist.shape, ref_hist.flatten().shape[0])

def init_particles(frame):
	rows, columns, channels = frame.shape

	# bounding boxes
	rows = rows // _BOX_HEIGHT
	columns = columns // _BOX_WIDTH

	# particledt = np.dtype([('myintname', np.int32), ('myfloats', np.float64, 9)])
	particles_matrix = np.empty((rows, columns), dtype=object)
	
	for i in range(rows):
		for j in range(columns):
			particles_matrix[i,j] = Particle(0, (i + 1) * _BOX_HEIGHT, j * _BOX_WIDTH)

	return particles_matrix

#Draw green particles
def draw_particles(frame, particles_matrix):
	for index, part_temp in np.ndenumerate(particles_matrix):
		cv2.circle(frame, (part_temp.q[0], part_temp.q[1]), 1, (0, 255, 0), -1)

#Draw tracks center
def draw_tracks(frame, tracks_arr):
	for track in tracks_arr:
		cv2.circle(frame, (track.p[0], track.p[1]), 2, (255, 0, 0), -1)

#Draw positive particles
def draw_pos_particles(frame, pparticles_set):
	for particle in pparticles_set:
		cv2.circle(frame, (particle.q[0], particle.q[1]), 1, (0, 255, 0), -1)

def findInnerParticles(particles_matrix, x, y, w, h):
	rows, columns = particles_matrix.shape

	xi_p = (x + _BOX_WIDTH - 1) //_BOX_WIDTH
	yi_p = y //_BOX_HEIGHT
	xf_p = (x + w) //_BOX_WIDTH
	yf_p = (y + h) //_BOX_HEIGHT - 1

	inner_particles = []
	
	for i in range(yi_p, yf_p + 1):
		for j in range(xi_p, xf_p + 1):
			if 0 <= i < rows and 0 <= j < columns:
				inner_particles.append(particles_matrix[i,j])

	return inner_particles


def calcColorProb(frame, s_temp, x_temp):
	
	lambd = 20

	s_temp.obtainHistogram(frame)

	d_sup = cv2.compareHist(s_temp.hist_sup, x_temp.R, cv2.HISTCMP_BHATTACHARYYA)
	d_inf = cv2.compareHist(s_temp.hist_inf, x_temp.R, cv2.HISTCMP_BHATTACHARYYA)

	d_prom = (d_sup**2 + d_inf**2) / 2
	return math.exp(-1*lambd*d_prom)

def calcMotionProb(frame, s_temp, x_temp):
	
	std_dev = 5
	dist.euclidean(x_temp.p, s_temp.q)

	first_factor = 1 / (std_dev * (math.pi**(0.5)))
	second_factor = math.exp(-1 * (dist.euclidean(s_temp.q, x_temp.p)**2) / (std_dev**2))

	return first_factor * second_factor

def dict_normalize(d):
	norm_d = {}
	factor = 1.0 / sum(d.itervalues())
	for key in d:
		norm_d[key] = d[key] * factor

	return norm_d