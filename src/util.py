from unittest import result
from cv2 import threshold
import open3d as o3d
import numpy as np
import os, glob, re
import pyk4a, cv2, json

from pyk4a import Config, PyK4A, PyK4APlayback
from scipy.spatial.transform import Rotation

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def export_intrinsic_calib(dir, json_fn = "intrinsic.json", mkv_fn="capture.mkv"):
	mkv_path = os.path.join(dir, mkv_fn)
	playback = pyk4a.PyK4APlayback(mkv_path)

	playback.open()

	tof_intrinsic_array = playback.calibration.get_camera_matrix(
		pyk4a.calibration.CalibrationType.DEPTH)
	tof_intrinsic_list = tof_intrinsic_array.flatten().tolist()

	tof_distortion_array = playback.calibration.get_distortion_coefficients(
		pyk4a.CalibrationType.DEPTH)
	tof_distortion_list = tof_distortion_array.flatten().tolist()

	data = {"intrinsic_matrix": tof_intrinsic_list, "distortion_coefficients": tof_distortion_list}
	
	json_path = os.path.join(dir, json_fn)
	
	with open(json_path, 'w') as f:
		json.dump(data, f)

	return tof_intrinsic_list, tof_distortion_list

def import_intrinsic_calib(dir, fn="intrinsic.json"):
	 # Create the full file path using os.path.join
    file_path = os.path.join(dir,fn)

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"The file {file_path} does not exist.")
        return None, None

    # Open and read the JSON file
    data = readJSON(file_path)

    # Extract the intrinsic matrix and distortion coefficients
    intrinsic_matrix = data.get("intrinsic_matrix", [])
    distortion_coefficients = data.get("distortion_coefficients", [])

    intrinsic_matrix = np.array(intrinsic_matrix)
    distortion_coefficients = np.array(distortion_coefficients)

    return np.reshape(intrinsic_matrix, (3,3)), distortion_coefficients

def parseMKV(dir = "./mkv_calib", exportDepth = False, debug = False):
	fn = "%s/capture.mkv" % dir
	if debug:
		print("Opening MKV file at %s..." % fn, end="")

	reader = o3d.io.AzureKinectMKVReader()
	reader.open(fn)

	if not reader.is_opened():
		raise RuntimeError("Failed!")
	else:
		if debug:
			print("Success!")

	if not os.path.exists("%s/color" % dir):
		os.mkdir("%s/color" % dir)
	if exportDepth:
		if not os.path.exists("%s/depth" % dir):
			os.mkdir("%s/depth" % dir)

	idx = 0
	while not reader.is_eof():
		rgbd = reader.next_frame()
		if rgbd is None:
			continue

		# save color image
		color_fn = "%s/color/%03d.jpg" % (dir,idx)
		if debug:
			print('Writing to {}'.format(color_fn))
		o3d.io.write_image(color_fn, rgbd.color)

		# save depth image
		if exportDepth:
			depth_fn = "%s/depth/%03d.png" % (dir,idx)
			if debug:
				print('Writing to {}'.format(depth_fn))
			o3d.io.write_image(depth_fn, rgbd.depth)

		idx += 1

def tcp2base(dir = "./mkv_calib", debug = False):
	fn = glob.glob("%s/*.mod" % dir)

	if len(fn) > 1:
		raise RuntimeError("More than 1 .mod files exist!")
	elif len(fn) == 0:
		raise RuntimeError("No .mod files exist!")
	else:
		fn = fn[0]

	file = open(fn, 'r')
	lines = file.readlines()

	posL = []
	quatL = []
	rVecL = []
	rMatL = []
	H = []
	for line in lines:
		tmp = line.split()
		if tmp[0] == "MoveL":
			tmp = tmp[1][1:-1].split(']')

			pos = tmp[0][1:].split(",")
			pos = [float(i) for i in pos]
			# pos[-1] = pos[-1] + 12 ########################################################################################################################

			quat = tmp[1][2:].split(",")
			quat = [float(i) for i in quat]
			quat = [quat[1],quat[2],quat[3],quat[0]]

			rMat = Rotation.from_quat(quat).as_matrix()
			rVec = Rotation.from_quat(quat).as_rotvec()
			posL.append(pos)
			quatL.append(quat)
			rVecL.append(rVec)
			rMatL.append(rMat)
			H.append(composeH(rMat,pos))
	t_tcp2base = np.asarray(posL).reshape(-1,3,1)
	q_tcp2base = np.asarray(quatL)
	rVec_tcp2base = np.asarray(rVecL)
	rMat_tcp2base = np.asarray(rMatL)
	H_tcp2base = np.asarray(H)

	if debug:
		print("Saving tcp2base...", end="")
	np.savez("%s/tcp2base" % dir, t=t_tcp2base, rVec=rVec_tcp2base, rMat=rMat_tcp2base, q=q_tcp2base, H=H_tcp2base)
	if debug:
		print("Done!")

	return t_tcp2base, rVec_tcp2base, rMat_tcp2base, q_tcp2base

def dumpTrajectoryLogs(dir, t_tcp2base, rMat_tcp2base, cam2tcp_color, methods):
	result = {}
	H_cam2base = {}
	for method in methods:
		if method not in result:
			result[method] = []
			H_cam2base[method] = []
		with open('%s/trajectory_%s.log' % (dir, method), 'w') as f:
			for i in range(len(t_tcp2base)):
				f.write('{} {} {}\n'.format(i-1, i, len(t_tcp2base)))

				T = composeH(rMat_tcp2base[i], t_tcp2base[i])@cam2tcp_color[method]
				H_cam2base[method].append(T)
				T[:3, 3] *= 0.001

				result[method].append(T)
				s = np.array2string(T)
				s = re.sub('[\[\]]', '', s)

				f.write('{}\n'.format(s))
	return result, H_cam2base

def target2cam_ir(dir = None, mtx = None, dist = None, rx=None, tx=None, boardSize = (9,6), boardDim = 25, debug = False, normalizeIR=True, thresh=None):
	size = None

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	objp = np.zeros((boardSize[0]*boardSize[1],3), np.float32)
	objp[:,:2] = np.mgrid[0:boardSize[0],0:boardSize[1]].T.reshape(-1,2)
	objp *= boardDim

	# axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
	# axis *= boardDim
	images = glob.glob("%s/ir/*.png" %dir)

	failedIdx = []
	rVec_target2cam = []
	rMat_target2cam = []
	t_target2cam = []

	for i,fname in enumerate(images):
		gray = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
		if thresh is not None:
			gray[np.where(gray>thresh)] = thresh
		if normalizeIR:
			gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

		if size is None:
			size = gray.shape[::-1]
		if debug:
			print("Estimating pose for image %3d..." % i,end="")

		ret, corners = cv2.findChessboardCornersSB(gray, boardSize, None, cv2.CALIB_CB_ACCURACY+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_EXHAUSTIVE)

		if ret:
			corners2 = cv2.cornerSubPix(gray, corners,(5,5),(-1,-1), criteria)
			# Find the rotation and translation vectors.
			ret, rVec, t = cv2.solvePnP(objp, corners2, mtx, dist)#, rvec=rx[i], tvec=tx[i])
			if ret:
				rVec, t = cv2.solvePnPRefineLM(objp, corners2, mtx, dist, rVec, t)
			rMat = Rotation.from_rotvec(rVec.reshape((-1,))).as_matrix()

			rVec_target2cam.append(rVec)
			rMat_target2cam.append(rMat)
			t_target2cam.append(t)
			# project 3D points to image plane
			# imgpts, _ = cv2.projectPoints(axis, rVec, t, mtx, dist)
			if debug:
				img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
				img = cv2.drawChessboardCorners(img, boardSize, corners, ret)
				cv2.imshow('img',img)
				cv2.waitKey(1)
				print("Done!")
		else:
			if debug:
				print("Failed!")
			failedIdx.append(i)
	if debug:
		if len(failedIdx) > 0:
			print("Failed to detect chessboard in %d images." % len(failedIdx))
			print(failedIdx)

	rVec_target2cam = np.asarray(rVec_target2cam)
	rMat_target2cam = np.asarray(rMat_target2cam)
	t_target2cam = np.asarray(t_target2cam).reshape(-1,3,1)

	if debug:
		print("Saving target2cam...", end="")
	np.savez("%s/target2cam" % dir, rVec=rVec_target2cam, rMat=rMat_target2cam, t=t_target2cam, failedIdx=failedIdx)
	if debug:
		cv2.destroyAllWindows()
		print("Done!")

	return rVec_target2cam, rMat_target2cam, t_target2cam, failedIdx

def target2cam_color(dir = None, mtx = None, dist = None, rx=None, tx=None, boardSize = (9,6), boardDim = 25, debug = False):
	size = None

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	objp = np.zeros((boardSize[0]*boardSize[1],3), np.float32)
	objp[:,:2] = np.mgrid[0:boardSize[0],0:boardSize[1]].T.reshape(-1,2)
	objp *= boardDim

	axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
	axis *= boardDim
	images = glob.glob("%s/color/*.jpg" %dir)

	failedIdx = []
	rVec_target2cam = []
	rMat_target2cam = []
	t_target2cam = []

	for i,fname in enumerate(images):
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		if size is None:
			size = gray.shape[::-1]
		if debug:
			print("Estimating pose for image %3d..." % i,end="")

		gray_small = cv2.resize(gray, None, fx=.5, fy=.5)
		ret, corners = cv2.findChessboardCorners(gray_small, boardSize,  None)

		if ret:
			ret, corners = cv2.findChessboardCorners(gray, boardSize,  None)
			corners2 = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1), criteria)
			# Find the rotation and translation vectors.
			ret, rVec, t = cv2.solvePnP(objp, corners2, mtx, dist, rvec=rx[i], tvec=tx[i])
			rVec, t = cv2.solvePnPRefineLM(objp, corners2, mtx, dist, rVec, t)
			rMat = Rotation.from_rotvec(rVec.reshape((-1,))).as_matrix()

			rVec_target2cam.append(rVec)
			rMat_target2cam.append(rMat)
			t_target2cam.append(t)
			# project 3D points to image plane
			# imgpts, _ = cv2.projectPoints(axis, rVec, t, mtx, dist)
			if debug:
				img = cv2.drawChessboardCorners(img, boardSize, corners2, ret)
				img = cv2.resize(img, None, fx=.25, fy=.25)
				cv2.imshow('img',img)
				cv2.waitKey(1)
				print("Done!")
		else:
			if debug:
				print("Failed!")
			failedIdx.append(i)
	if debug:
		if len(failedIdx) > 0:
			print("Failed to detect chessboard in %d images." % len(failedIdx))
			print(failedIdx)

	rVec_target2cam = np.asarray(rVec_target2cam)
	rMat_target2cam = np.asarray(rMat_target2cam)
	t_target2cam = np.asarray(t_target2cam).reshape(-1,3,1)

	if debug:
		print("Saving target2cam...", end="")
	np.savez("%s/target2cam" % dir, rVec=rVec_target2cam, rMat=rMat_target2cam, t=t_target2cam, failedIdx=failedIdx)
	if debug:
		cv2.destroyAllWindows()
		print("Done!")

	return rVec_target2cam, rMat_target2cam, t_target2cam, failedIdx

def target2cam_depth(dir = None, trajectory = None, methods = None, threshold = 1.0/512, debug = False):
	images = glob.glob("%s/depth/*.png" %dir)
	ret = {}
	fit = {}
	pinhole_extrinsics = {}
	for method in methods:
		data = o3d.io.read_pinhole_camera_trajectory('%s/trajectory_%s.log' % (dir, method))
		extr = []
		for param in data.parameters:
			extr.append(param.extrinsic)
		assert(len(images) == len(extr))
		pinhole_extrinsics[method] = extr
	target = o3d.io.read_point_cloud('%s\\target.ply' % dir)

	intrinsic = o3d.io.read_pinhole_camera_intrinsic('%s/intrinsic.json' % dir)
	for i in range(len(images)):
		img = o3d.io.read_image(images[i])
		for method in methods:
			if method not in ret:
				ret[method] = []
				fit[method] = []
			extrinsic = pinhole_extrinsics[method][i]

			pcd = o3d.geometry.PointCloud.create_from_depth_image(img, intrinsic, extrinsic)

			points = np.asarray(pcd.points)
			maskY = np.ma.masked_inside(points[:,1], -0.275, 0.275).mask
			maskX = np.ma.masked_inside(points[:,0], 0.725, 1.275).mask
			mask = np.where(np.logical_and(maskX,maskY))
			source = pcd.select_by_index(mask[0])
			o3d.io.write_point_cloud('./test.ply', source)

			guess = np.eye(4)
			reg_p2l = o3d.pipelines.registration.registration_icp(source, target, threshold, guess,
																  o3d.pipelines.registration.TransformationEstimationPointToPlane(),
																  o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
			T = reg_p2l.transformation
			T = np.linalg.inv(T)

			est_pose = np.linalg.inv(pinhole_extrinsics[method][i]@T)
			est_pose[:3,3] *= 1000

			if debug:
				print("-----Pose #%d | Method:%s-----" % (i, method))
				print(reg_p2l)
				print("Transformation:")
				print(T)
				print("Input Pose:")
				print(np.linalg.inv(pinhole_extrinsics[method][i]))
				print("Estimated Pose:")
				print(np.linalg.inv(pinhole_extrinsics[method][i]@T))
			
			fit[method].append(reg_p2l.fitness)
			ret[method].append(est_pose)
	np.save("%s/est_pose" % dir, ret)
	np.save("%s/fitness" % dir, fit)
	np.save("%s/pinhole_extr" % dir, pinhole_extrinsics)

	return ret, fit, pinhole_extrinsics

def deconstructPoses(est_pose, scale = 1.0):
	rVec_pose = []
	t_pose = []
	for pose in est_pose:
		t = pose[:3,3]
		t = t.reshape(-1,3,1)
		# t *= scale
		t_pose.append(t)

		rVec = Rotation.from_matrix(pose[:3,:3]).as_rotvec()
		rVec_pose.append(rVec)
	return t_pose, rVec_pose

def composeH(R, t):
	H = np.eye(4)
	H[:3, :3] = np.asarray(R)
	H[:3, 3] =  np.asarray(t).ravel()
	return np.asarray(H)

def decomposeH(H):
	R = H[:3, :3]
	t = H[:3, 3]
	return R, t

def writeJSON(data, file_path):
	print(f"Writing JSON file {file_path} ... ", end='')
	with open(file_path, 'w') as f:
		json.dump(data, f)
	print("Done")
	f.close

def readJSON(file_path):
	print(f"Reading JSON file {file_path} ... ", end='')
	with open(file_path, 'r') as f:
		data = json.load(f)
	f.close()
	print("Done")
	return data