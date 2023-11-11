import os
import cv2
import pyk4a
from src.util import writeJSON


class AzureKinectMKVParser(object):
	"""
	A parser class for processing MKV files captured using an Azure Kinect device.
	This class allows for the extraction of color, depth, and infrared frames, 
	as well as exporting calibration data from the MKV file.
	"""

	def __init__(
		self,
		dir,
		file_name='capture.mkv',
	) -> None:
		"""
		Initializes the AzureKinectMKVParser object.

		:param dir: The directory where the MKV file is located.
		:param file_name: The name of the MKV file to be parsed. Default is 'capture.mkv'.
		"""
		self.dir = dir
		self.mkv_path = os.path.join(dir, file_name)
		self.playback = pyk4a.PyK4APlayback(self.mkv_path)

	def export_frames(self, color=True, depth=True, ir=True):
		"""
		Exports color, depth, and infrared frames from the MKV file.

		:param color: A boolean indicating if color frames should be exported. Default is True.
		:param depth: A boolean indicating if depth frames should be exported. Default is True.
		:param ir: A boolean indicating if infrared frames should be exported. Default is True.
		"""
		# Directory paths for color, depth, and IR images
		color_path = os.path.join(self.dir, 'color')
		depth_path = os.path.join(self.dir, 'depth')
		ir_path = os.path.join(self.dir, 'ir')

        # Creating directories for storing the exported frames
		if color and not os.path.exists(color_path):
			os.mkdir(color_path)
		if depth and not os.path.exists(depth_path):
			os.mkdir(depth_path)
		if ir and not os.path.exists(ir_path):
			os.mkdir(ir_path)

		self.playback.open()  # Open the MKV file for reading

		i = 0
		while True:
			try:
				capture = self.playback.get_next_capture()
                # Write the frames to the respective directories as image files
				if color and capture.color is not None:
					path = os.path.join(color_path, f'color_{i:03}.png')
					cv2.imwrite(path, capture.color)
				if depth and capture.depth is not None:
					path = os.path.join(depth_path, f'depth_{i:03}.png')
					cv2.imwrite(path, capture.depth)
				if ir and capture.ir is not None:
					path = os.path.join(ir_path, f'ir_{i:03}.png')
					cv2.imwrite(path, capture.ir)
				i += 1
			except EOFError:
				break  # End of file reached

		self.playback.close()  # Close the MKV file after exporting frames

	def export_calibration(self):
		"""
		Exports the intrinsic calibration matrix and distortion coefficients from the MKV file.

		:return: A tuple containing lists of intrinsic matrix and distortion coefficients.
		"""
		self.playback.open()  # Open the MKV file for reading

        # Retrieve intrinsic calibration and distortion data from the MKV file
		tof_intrinsic_array = self.playback.calibration.get_camera_matrix(
			pyk4a.calibration.CalibrationType.DEPTH)
		tof_intrinsic_list = tof_intrinsic_array.tolist()

		tof_distortion_array = self.playback.calibration.get_distortion_coefficients(
			pyk4a.CalibrationType.DEPTH)
		tof_distortion_list = tof_distortion_array.tolist()

		# Writing calibration data to a JSON file for later use
		data = {"intrinsic_matrix": tof_intrinsic_list,
				"distortion_coefficients": tof_distortion_list}

		json_path = os.path.join(self.dir, 'intrinsic.json')
		writeJSON(data, json_path)  # Write calibration data to a JSON file

		return tof_intrinsic_list, tof_distortion_list
