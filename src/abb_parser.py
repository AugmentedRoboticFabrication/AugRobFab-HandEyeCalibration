import glob, re, os, json
import numpy as np
from scipy.spatial.transform import Rotation
from src.util import composeH, decomposeH, readJSON, writeJSON

class T_ROBParser:
	def __init__(
			self, dir = None,
			extrinsic = None,
			parsing_method = 'robtarget'
			) -> None:
		
		self.dir = dir

		self.parsing_method = parsing_method.lower()

		if extrinsic is not None: # Convert to JSON
			extrinsic_dir = os.path.join(self.dir, extrinsic)
			data = readJSON(extrinsic_dir)
			self.extrinsic = data.get('extrinsic')
		else:
			self.extrinsic = extrinsic

	def _find_mod_file(self):
		dir = os.path.join(self.dir, '*.mod')
		mod_files = glob.glob(dir)
		if len(mod_files) > 1:
			raise RuntimeError("More than 1 .mod files exist!")
		elif len(mod_files) == 0:
			raise RuntimeError("No .mod files exist!")
		else:
			return mod_files[0]
	
	def _parse_mod_file(self, file_name):
		tmp = line.split()
		if tmp[0] == "MoveL":
			tmp = tmp[1][1:-1].split(']')

			pos = tmp[0][1:].split(",")
			pos = [float(i) for i in pos]
			pos = np.asarray(pos).reshape(-1,3)

			quat = tmp[1][2:].split(",")
			quat = [float(i) for i in quat]
			quat = [quat[1],quat[2],quat[3],quat[0]]
		pass

	def _parse_robtarget_line(self, line):
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
			quat = [quat[1],quat[2],quat[3],quat[0]]

			ret = True
		return ret, pos, quat
	
	def _parse_moveL_line(self, line):
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
			quat = [quat[1],quat[2],quat[3],quat[0]]

			ret = True
		return ret, pos, quat
	
	def base_T_tcp(self):
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
				raise RuntimeError('Parsing method is not valid! Only robtarget or movel')
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
			return np.asarray(L_pos), np.asarray(L_quat), np.asarray(L_rotmat), np.asarray(L_rotvec), np.asarray(L_H)
		else:
			raise RuntimeError(f"No robtarget or MoveL keywords were found in {mod_file}!")
		
	def base_T_camera(self):
		if self.extrinsic is None:
			raise RuntimeError('No extrinsic was given!')
		_, _, _, _, Ts = self.base_T_tcp()
		
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
			return np.asarray(L_pos), np.asarray(L_quat), np.asarray(L_rotmat), np.asarray(L_rotvec), np.asarray(L_H)


		return base_T_camera
		
	def trajectory(self):
		
		Ts = self.base_T_camera()
		n = len(Ts)
		result = []

		path = os.path.join(self.dir, 'trajectory.log')
		with open(path, 'w') as f:
			for i in range(n):
				f.write('{} {} {}\n'.format(i-1, i, n))

				T = Ts[i]
				T[:3, 3] *= 0.001

				result.append(T)
				s = np.array2string(T)
				s = re.sub('[\[\]]', '', s)

				f.write('{}\n'.format(s))
		return result