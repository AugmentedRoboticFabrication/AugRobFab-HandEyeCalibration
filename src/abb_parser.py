import glob
import re
import os
import numpy as np
from scipy.spatial.transform import Rotation
from src.util import composeH, decomposeH, readJSON, writeJSON


class TROBParser:
	"""
	A parser class for processing robotic data, specifically for extracting robot trajectories
	and transformations from .mod files and calculating transformations between different frames.
	"""

	def __init__(
		self,
		dir=None,
		extrinsic_dir=None,
		parsing_method='robtarget'
	) -> None:
		"""
		Initializes the TROBParser object.

		:param dir: Directory where the .mod files and other related data are stored.
		:param extrinsic: Path to the JSON file containing extrinsic calibration data.
		:param parsing_method: Method used for parsing the .mod files. Default is 'robtarget'.
		"""

		self.dir = dir

		self.parsing_method = parsing_method.lower()

		if extrinsic_dir is not None:
			data = readJSON(extrinsic_dir)
			self.extrinsic = np.asarray(data.get('extrinsic')).reshape((4,4))
			print("-------------------")
			print("Extrinsic Calibration:")
			print(self.extrinsic)
			print("-------------------")
		else:
			self.extrinsic = extrinsic_dir

	def _find_mod_file(self):
		"""
		Locates the .mod file in the specified directory.

		:return: Path to the .mod file.
		:raises RuntimeError: If more than one or no .mod file is found.
		"""

		dir = os.path.join(self.dir, '*.mod')
		mod_files = glob.glob(dir)
		if len(mod_files) > 1:
			raise RuntimeError("More than 1 .mod files exist!")
		elif len(mod_files) == 0:
			raise RuntimeError("No .mod files exist!")
		else:
			return mod_files[0]

	def _parse_mod_file(self, file_name):
		"""
		Parses the given .mod file for robot movement instructions.

		:param file_name: Name of the .mod file to be parsed.
		"""

		tmp = line.split()
		if tmp[0] == "MoveL":
			tmp = tmp[1][1:-1].split(']')

			pos = tmp[0][1:].split(",")
			pos = [float(i) for i in pos]
			pos = np.asarray(pos).reshape(-1, 3)

			quat = tmp[1][2:].split(",")
			quat = [float(i) for i in quat]
			quat = [quat[1], quat[2], quat[3], quat[0]]
		pass

	def _parse_robtarget_line(self, line):
		"""
		Parses a single line from the .mod file for robtarget instructions.

		:param line: A line from the .mod file.
		:return: A tuple (ret, pos, quat) where ret is a boolean indicating if the line contains valid data,
				 pos is the position vector, and quat is the quaternion for orientation.
		"""

		pos = None
		quat = None
		ret = False

		tmp = line.split()

		if len(tmp) > 1 and tmp[1] == "robtarget":
			tmp = tmp[2].split(']')
			tmp = [i.split('[')[-1] for i in tmp]

			pos = tmp[0].split(",")
			pos = [float(i) for i in pos]

			quat = tmp[1].split(",")
			quat = [float(i) for i in quat]
			quat = [quat[1], quat[2], quat[3], quat[0]]

			ret = True
		return ret, pos, quat

	def _parse_moveL_line(self, line):
		"""
		Parses a single line from the .mod file for MoveL instructions.

		:param line: A line from the .mod file.
		:return: A tuple (ret, pos, quat) as in _parse_robtarget_line.
		"""

		pos = None
		quat = None
		ret = False

		tmp = line.split()

		if len(tmp) > 1 and tmp[0] == "MoveL":
			tmp = tmp[1][1:-1].split(']')

			pos = tmp[0][1:].split(",")
			pos = [float(i) for i in pos]

			quat = tmp[1][2:].split(",")
			quat = [float(i) for i in quat]
			quat = [quat[1], quat[2], quat[3], quat[0]]

			ret = True
		return ret, pos, quat

	def base_T_tcp(self):
		"""
		Calculates the transformations from the base to the tool center point (TCP) from the .mod file.

		:return: Arrays of positions, quaternions, rotation matrices, rotation vectors, and homogeneous matrices.
		:raises RuntimeError: If no valid robtarget or MoveL keywords are found.
		"""
		
		mod_path = self._find_mod_file()

		file = open(mod_path, 'r')
		lines = file.readlines()

		L_pos = []
		L_quat = []
		L_rotmat = []
		L_rotvec = []
		L_H = []
		for line in lines:
			if self.parsing_method == 'robtarget':
				ret, pos, quat = self._parse_robtarget_line(line)
			elif self.parsing_method == 'movel':
				ret, pos, quat = self._parse_moveL_line(line)
			else:
				raise RuntimeError(
					'Parsing method is not valid! Only robtarget or movel')
			if ret:
				L_pos.append(pos)
				L_quat.append(quat)

				tmp = Rotation.from_quat(quat)
				L_rotmat.append(tmp.as_matrix().tolist())
				L_rotvec.append(tmp.as_rotvec().tolist())
				L_H.append(composeH(tmp.as_matrix(), pos).tolist())

		if len(L_pos) > 0:
			data = {
				'pos': L_pos,
				'rotvec': L_rotvec,
				'rotmat': L_rotmat,
				'quat': L_quat,
				'H': L_H
			}
			path = os.path.join(self.dir, 'base_T_tcp.json')
			writeJSON(data, path)
			return data
		else:
			raise RuntimeError(
				f"No robtarget or MoveL keywords were found in {mod_file}!")

	def base_T_camera(self):
		"""
		Calculates the transformations from the base to the camera frame.

		:return: Arrays of positions, quaternions, rotation matrices, rotation vectors, and homogeneous matrices.
		:raises RuntimeError: If no extrinsic calibration data is provided.
		"""

		if self.extrinsic is None:
			raise RuntimeError('No extrinsic was given!')
		data = self.base_T_tcp()
		Ts = np.asarray(data.get('H')).reshape((-1,4,4))

		L_pos = []
		L_quat = []
		L_rotmat = []
		L_rotvec = []
		L_H = []

		base_T_camera = []
		for T in Ts:
			tmp = T@self.extrinsic
			base_T_camera.append(tmp)

			rotmat, pos = decomposeH(tmp)
			L_pos.append(pos.tolist())
			L_rotmat.append(rotmat.tolist())
			tmp = Rotation.from_matrix(rotmat)

			L_quat.append(tmp.as_quat().tolist())
			L_rotvec.append(tmp.as_rotvec().tolist())
			L_H.append(composeH(tmp.as_matrix(), pos).tolist())

		if len(L_pos) > 0:
			data = {
				'pos': L_pos,
				'rotvec': L_rotvec,
				'rotmat': L_rotmat,
				'quat': L_quat,
				'H': L_H
			}
			path = os.path.join(self.dir, 'base_T_camera.json')
			writeJSON(data, path)
			return data
		
	def trajectory(self):
		"""
		Computes and logs the robot's trajectory based on the transformations to the camera frame.

		:return: A list of transformation matrices representing the robot's trajectory.
		"""

		data = self.base_T_camera()
		Ts = np.asarray(data.get('H')).reshape((-1,4,4))
		n = len(Ts)
		result = []

		path = os.path.join(self.dir, 'trajectory.log')
		with open(path, 'w') as f:
			for i in range(n):
				f.write('{} {} {}\n'.format(i-1, i, n))

				T = np.asarray(Ts[i]).reshape((4,4))
				
				T[:3, 3] *= 0.001

				result.append(T)
				s = np.array2string(T)
				s = re.sub('[\[\]]', '', s)

				f.write('{}\n'.format(s))
		return result
