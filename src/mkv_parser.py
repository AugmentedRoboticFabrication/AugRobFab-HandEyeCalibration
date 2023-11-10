import os, cv2
import pyk4a
import numpy as np

class AzureKinectMKVParser(object):
	def __init__(
			self,
			dir,
			file_name='capture.mkv',
			):
		self.dir = dir
		
		self.mkv_path = os.path.join(dir, file_name)
		self.playback = pyk4a.PyK4APlayback(self.mkv_path)

	def export_frames(self, color=True, depth=True, ir=True):
		color_path = os.path.join(self.dir, 'color')
		if color and not os.path.exists(color_path):
			os.mkdir(color_path)

		depth_path = os.path.join(self.dir, 'depth')
		if depth and not os.path.exists(color_path):
			os.mkdir(depth_path)

		ir_path = os.path.join(self.dir, 'ir')
		if ir and not os.path.exists(color_path):
			os.mkdir(ir_path)

		self.playback.open()

		i = 0
		while True:
			try:
				capture = self.playback.get_next_capture()
				if exportColor and capture.color is not None:
					path = os.path.join(color_path, f'color_{i:03}.png')
					cv2.imwrite(path, capture.color)
				if exportDepth and capture.depth is not None:
					path = os.path.join(depth_path, f'depth_{i:03}.png')
					cv2.imwrite(path, capture.depth)				
				if exportIR and capture.ir is not None:
					path = os.path.join(ir_path, f'ir_{i:03}.png')
					cv2.imwrite(path, capture.ir)
			except EOFError:
				break
		
		self.playback.close()

		# self.reader.open(self.mkv)

		# metadata = self.reader.get_metadata()
		# o3d.io.write_azure_kinect_mkv_metadata('%s\\%s\\intrinsic.json' %(self.root, self.dir), metadata)

		# index = 0
		# while not self.reader.is_eof():
		# 	rgbd = self.reader.next_frame()
		# 	if rgbd is None:
		# 		continue

		# 	if color:
		# 		color_fn = '%s\\%s\\color\\%03d.jpg' % (self.root, self.dir, index)
		# 		o3d.io.write_image(color_fn, rgbd.color)

		# 	if depth:
		# 		depth_fn = '%s\\%s\\depth\\%03d.png' % (self.root, self.dir, index)
		# 		o3d.io.write_image(depth_fn, rgbd.depth)
		# 	index += 1
		
		# self.reader.close()

	def calibration(self):
		os.add_dll_directory("C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin")
		import pyk4a
		from pyk4a import PyK4APlayback

		self.playback = PyK4APlayback(self.mkv)
		print('Exporting factory calibration:')
		self.playback.open()

		mtx = self.playback.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR).tolist()
		dist = self.playback.calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR).tolist()

		np.savez('%s\\%s\\calibration' %(self.root, self.dir), mtx=mtx, dist=dist)

		print('Camera Matrix:\n', mtx)
		print('Distortion Coefficients:\n',dist)

		self.playback.close()