import numpy as np
import argparse
import imutils
import cv2

_BOX_WIDTH = 4
_BOX_HEIGHT = 6

class Particle:

	def __init__(self, weight, y, x, hist = None):
		self.weight = weight
		self.x = x
		self.y = y
		# self.scale = scale
		self.hist = hist

class Track:

	def __init__(self, y, x, v = 0):
		self.x = x
		self.y = y
		self.v = v

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
		cv2.circle(frame, (part_temp.x, part_temp.y), 1, (0, 255, 0), -1)

#Draw tracks center
def draw_tracks(frame, tracks_arr):
	for track in tracks_arr:
		cv2.circle(frame, (track.x, track.y), 2, (255, 0, 0), -1)

#Draw positive particles
def draw_pos_particles(frame, pparticles_set):
	for particle in pparticles_set:
		cv2.circle(frame, (particle.x, particle.y), 1, (0, 255, 0), -1)

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
