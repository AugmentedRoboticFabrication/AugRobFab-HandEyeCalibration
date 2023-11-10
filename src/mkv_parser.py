import os
import open3d as o3d
import numpy as np

class AzureKinectMKVParser(object):
	def __init__(self, fn, root=None):
		self.fn = fn

		if root is None:
			self.root = os.getcwd()
		else:
			self.root = root
		
		self.mkv = '%s\\%s\\capture.mkv' %(self.root, self.fn)
		self.playback = None
		self.reader = o3d.io.AzureKinectMKVReader()

	def rgbd(self, color=False, depth=False):
		# frames = []

		if color and not os.path.exists('%s\\%s\\color' %(self.root, self.fn)):
			os.mkdir('%s\\%s\\color' %(self.root, self.fn))
		
		if depth and not os.path.exists('%s\\%s\\depth' %(self.root, self.fn)):
			os.mkdir('%s\\%s\\depth' %(self.root, self.fn))

		self.reader.open(self.mkv)

		metadata = self.reader.get_metadata()
		o3d.io.write_azure_kinect_mkv_metadata('%s\\%s\\intrinsic.json' %(self.root, self.fn), metadata)

		index = 0
		while not self.reader.is_eof():
			rgbd = self.reader.next_frame()
			if rgbd is None:
				continue
			# frames.append(rgbd)

			if color:
				color_fn = '%s\\%s\\color\\%03d.jpg' % (self.root, self.fn, index)
				o3d.io.write_image(color_fn, rgbd.color)

			if depth:
				depth_fn = '%s\\%s\\depth\\%03d.png' % (self.root, self.fn, index)
				o3d.io.write_image(depth_fn, rgbd.depth)
			index += 1
		
		self.reader.close()

	def calibration(self):
		os.add_dll_directory("C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin")
		import pyk4a
		from pyk4a import PyK4APlayback

		self.playback = PyK4APlayback(self.mkv)
		print('Exporting factory calibration:')
		self.playback.open()

		mtx = self.playback.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR).tolist()
		dist = self.playback.calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR).tolist()

		np.savez('%s\\%s\\calibration' %(self.root, self.fn), mtx=mtx, dist=dist)

		print('Camera Matrix:\n', mtx)
		print('Distortion Coefficients:\n',dist)

		self.playback.close()