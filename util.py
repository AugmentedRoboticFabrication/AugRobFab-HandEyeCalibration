import open3d as o3d
import numpy as np
import os, json, glob
import pyk4a, cv2

from pyk4a import Config, PyK4A, PyK4APlayback
from scipy.spatial.transform import Rotation

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def exportCalib(dir = "./mkv_calib", debug = False):
	playback = PyK4APlayback("%s/capture.mkv" % dir)
	playback.open()

	mtx = playback.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR).tolist()
	dist = playback.calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR).tolist()

	np.savez("%s/calibration" % dir, mtx=mtx, dist=dist)

	if debug:
		print("Camera Matrix:\n", mtx)
		print("Distortion Coefficients:\n",dist)

	playback.close()

	return np.asarray(mtx), np.asarray(dist)

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
	for line in lines:
		tmp = line.split()
		if tmp[0] == "MoveL":
			tmp = tmp[1][1:-1].split(']')
			
			pos = tmp[0][1:].split(",")
			quat = tmp[1][2:].split(",")

			pos = [float(i) for i in pos]
			quat = [float(i) for i in quat]
			quat = [quat[1],quat[2],quat[3],quat[0]]
			rMat = Rotation.from_quat(quat).as_matrix()
			rVec = Rotation.from_quat(quat).as_rotvec()
			# print(pos, quat, rVec, rMat)
			posL.append(pos)
			quatL.append(quat)
			rVecL.append(rVec)
			rMatL.append(rMat)
	t_tcp2base = np.asarray(posL).reshape(-1,3,1)
	q_tcp2base = np.asarray(quatL)
	rVec_tcp2base = np.asarray(rVecL)
	rMat_tcp2base = np.asarray(rMatL)

	if debug:
		print("Saving tcp2base...", end="")
	np.savez("%s/tcp2base" % dir, t=t_tcp2base, rVec=rVec_tcp2base, rMat=rMat_tcp2base, q=q_tcp2base)
	if debug:
		print("Done!")

	return t_tcp2base, rVec_tcp2base, rMat_tcp2base, q_tcp2base

def target2cam(dir = "./mkv_calib", mtx = None, dist = None, rx=None, tx=None, boardSize = (9,6), boardDim = 25, debug = False):
	size = None

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	objp = np.zeros((boardSize[0]*boardSize[1],3), np.float32)
	objp[:,:2] = np.mgrid[0:boardSize[0],0:boardSize[1]].T.reshape(-1,2)
	objp *= boardDim

	axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
	axis *= boardDim
	images = glob.glob("%s/color/*.jpg" %dir)
	
	failCount = 0
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
		
		ret, corners = cv2.findChessboardCorners(gray, boardSize,None)

		if ret == True:
			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)
			# Find the rotation and translation vectors.
			ret, rVec, t = cv2.solvePnP(objp, corners, mtx, dist, rvec=rx[i], tvec=tx[i])
			rVec, t = cv2.solvePnPRefineLM(objp, corners, mtx, dist, rVec, t)
			rMat = Rotation.from_rotvec(rVec.reshape((-1,))).as_matrix()

			rVec_target2cam.append(rVec)
			rMat_target2cam.append(rMat)
			t_target2cam.append(t)
			# project 3D points to image plane
			imgpts, _ = cv2.projectPoints(axis, rVec, t, mtx, dist)
			if debug:
				img = cv2.drawChessboardCorners(img, boardSize, corners2, ret)
				img = cv2.resize(img, None, fx=.25, fy=.25)
				cv2.imshow('img',img)
				cv2.waitKey(1)
				print("Done!")
		else:
			if debug:
				print("Failed!")
			failCount+=1
	if debug:
		if failCount > 0:
			print("Failed to detect chessboard in %d images." % failCount)
	
	rVec_target2cam = np.asarray(rVec_target2cam)
	rMat_target2cam = np.asarray(rMat_target2cam)
	t_target2cam = np.asarray(t_target2cam).reshape(-1,3,1)

	if debug:
		print("Saving target2cam...", end="")
	np.savez("%s/target2cam" % dir, rVec=rVec_target2cam, rMat=rMat_target2cam, t=t_target2cam)
	if debug:
		cv2.destroyAllWindows()
		print("Done!")

	return rVec_target2cam, rMat_target2cam, t_target2cam

def combineRT(R, t, inv=False):
	M = np.eye(4)
	if inv:
		R = Rotation.from_matrix(np.asarray(R)).inv().as_matrix()
		t *= -1
	M[:3, :3] = np.asarray(R)
	M[:3, 3] =  np.asarray(t).ravel()
	return M