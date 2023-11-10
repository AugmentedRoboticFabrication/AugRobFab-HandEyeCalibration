import glob, re, os
import numpy as np
from scipy.spatial.transform import Rotation
from src.util import composeH

class T_ROBParser:
	def __init__(
			self, folder_name=None, root=None,
			tcp_T_cam=None, parsing_method = 'robtarget'):
		
		self.folder_name = folder_name
		self.root = root

		self.parsing_method = parsing_method

		if tcp_T_cam is not None:
			tcp_T_cam_dir = os.path.join = (self.root, tcp_T_cam)
			self.tcp_T_cam = np.load(tcp_T_cam_dir)
		else:
			self.tcp_T_cam = np.eye(4)
		print('tcp_T_cam:')
		print(self.tcp_T_cam)
		print('----------------')

	def test(self):
		mod_file = self._find_mod_file()

		file = open(mod_file, 'r')
		lines = file.readlines()

		for line in lines:
			if self.parsing_method == 'robtarget':
				ret, pos, quat = self._parse_robtarget_line(line)
				if ret:
					print(pos, quat)
			elif self.parsing_method == 'MoveL':
				pass
			else:
				raise RuntimeError('Parsing method is not valid!')


		return mod_file

	def _find_mod_file(self):
		dir = os.path.join(self.root, self.folder_name, '*.mod')
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
		pass

	def tcp2base(self, export=True):
		file_name = self._find_mod_file()

		file = open(file_name, 'r')
		lines = file.readlines()
		
		result = []
		for line in lines:
			tmp = line.split()
			if len(tmp) > 1 and tmp[1] == "robtarget":
				tmp = tmp[2].split(']')
				tmp = [i.split('[')[-1] for i in tmp]

				t = tmp[0].split(",")
				t = [float(i) for i in t]
				t = np.asarray(t).reshape(-1,3)

				quat = tmp[1].split(",")
				quat = [float(i) for i in quat]
				quat = [quat[1],quat[2],quat[3],quat[0]]
				
				r = Rotation.from_quat(quat).as_matrix()
				r = np.asarray(r)

		if export:
			np.save("%s\\%s\\tcp2base" % (self.root, self.file_name), result)

		return result
	
	def cam2base(self, export=True):
		if self.cam2tcp is None:
			raise RuntimeError("No cam2tcp tranformation given!")

		tcp2base = self.tcp2base(export=True)
		
		result = []
		for i in range(len(tcp2base)):
			result.append(tcp2base[i]@self.cam2tcp)
		if export:
			np.save("%s\\%s\\cam2base" % (self.root, self.file_name), result)
		return result
		
	def trajectory(self):
		
		Ts = self.cam2base(export=True)
		n = len(Ts)
		result = []
		with open('%s\\%s\\trajectory.log' % (self.root, self.file_name), 'w') as f:
			for i in range(len(Ts)):
				f.write('{} {} {}\n'.format(i-1, i, n))

				T = Ts[i]
				T[:3, 3] *= 0.001

				result.append(T)
				s = np.array2string(T)
				s = re.sub('[\[\]]', '', s)

				f.write('{}\n'.format(s))
		return result