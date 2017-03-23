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