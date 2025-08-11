import os
import glob
import cv2
import pyk4a

import numpy as np

from scipy.spatial.transform import Rotation
from src.util import composeH, decomposeH, readJSON, writeJSON


class CameraCalibration:
    """
    A class for performing camera calibration using a checkerboard pattern.
    It detects the pattern in a set of images and calculates the camera's intrinsic and extrinsic parameters.
    """

    def __init__(
            self,
            dir,
            folder_name,
            intrinsic_matrix,
            distortion_coefficients,
            checkerboard_shape,
            checkerboard_size,
            save_dir='calib',
            ir_treshold=None,
            debug=False
    ) -> None:
        """
        Initializes the CameraCalibration object with necessary parameters for performing calibration.

        :param dir: Directory containing the images for calibration.
        :param folder_name: Name of the folder within 'dir' containing the calibration images.
        :param intrinsic_matrix: The intrinsic matrix of the camera.
        :param distortion_coefficients: The distortion coefficients of the camera.
        :param checkerboard_shape: The number of corners in the checkerboard pattern (rows, columns).
        :param checkerboard_size: The size of each square in the checkerboard pattern.
        :param save_dir: Directory where calibration results will be saved, default is 'calib'.
        :param ir_treshold: Infrared threshold for image processing, default is None.
        :param debug: If True, additional debugging information will be displayed, default is False.
        """

        self.dir = dir
        self.save_dir = save_dir
        self.folder_name = folder_name

        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coefficients = distortion_coefficients

        self.ir_treshold = ir_treshold

        self.checkerboard_shape = checkerboard_shape
        self.checkerboard_size = checkerboard_size

        self.debug = debug

    def _target_T_camera(self):
        """
        Detects a checkerboard in images and calculates the transformation from the checkerboard to the camera.
        This is used for calculating the extrinsic parameters of the camera.

        :return: Rotation vectors, translation vectors, and indices of failed detections.
        :raises RuntimeError: If no images are found in the specified directory.
        """

        # Object points, image paths, and other preparations
        objp = np.zeros(
            (self.checkerboard_shape[0]*self.checkerboard_shape[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_shape[0],
                               0:self.checkerboard_shape[1]].T.reshape(-1, 2)
        objp *= self.checkerboard_size

        image_path = os.path.join(self.dir, self.folder_name,  '*.png')
        images = glob.glob(image_path)

        if len(images) == 0:
            raise RuntimeError(
                f'No images found under {self.dir}/{self.folder_name}!')

        L_failed_id = []
        L_rotvec = []
        L_pos = []

        # Loop through images to detect checkerboard and calculate transformations
        for i, file_name in enumerate(images):
            im = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

            if len(im.shape) == 2:
                criteria = cv2.CALIB_CB_ACCURACY + \
                    cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_EXHAUSTIVE

                gray = im
                if self.ir_treshold is not None:
                    gray[np.where(gray > self.ir_treshold)] = self.ir_treshold
                gray = cv2.normalize(gray, None, 0, 255,
                                     cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                criteria = None

                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCornersSB(
                gray,
                self.checkerboard_shape,
                None,
                criteria,
            )

            if ret:
                corners2 = cv2.cornerSubPix(
                    gray,
                    corners,
                    (5, 5),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                ret, rVec, t = cv2.solvePnP(
                    objp, corners2, self.intrinsic_matrix, self.distortion_coefficients)
                if ret:
                    # If checkerboard is detected, calculate the rotation and translation vectors
                    rVec, t = cv2.solvePnPRefineLM(
                        objp,
                        corners2,
                        self.intrinsic_matrix,
                        self.distortion_coefficients,
                        rVec,
                        t,
                        )
                rMat = Rotation.from_rotvec(rVec.reshape((-1,))).as_matrix()

                L_rotvec.append(rVec)
                L_pos.append(t)
                if self.debug:
                    # Display the checkerboard corners in debug mode
                    im = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    im = cv2.drawChessboardCorners(
                        im, self.checkerboard_shape, corners, ret)
                    cv2.imshow('Image', im)
                    cv2.waitKey(1)
            else:
                # If checkerboard detection fails, log the image index
                L_failed_id.append(i)
                if self.debug:
                    im = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    cv2.imshow('Image', im)
                    cv2.waitKey(1)

        cv2.destroyAllWindows()

        if self.debug:
            if len(L_failed_id) > 0:
                print("Failed to detect chessboard in %d images." %
                      len(L_failed_id))
                print(L_failed_id)

        L_rotvec = np.asarray(L_rotvec)
        L_pos = np.asarray(L_pos).reshape(-1, 3, 1)

        return L_rotvec, L_pos, L_failed_id

    def calibrate(self, method=cv2.CALIB_HAND_EYE_TSAI, fn="tcp_T_camera.json"):
        """
        Performs hand-eye calibration using the detected checkerboard in images.
        This method is used to estimate the transformation between the robot's hand (end effector) and the camera.

        :param method: The calibration method to be used, default is cv2.CALIB_HAND_EYE_TSAI.
        :return: The estimated transformation matrix.
        """

        # Detect checkerboard in images and calculate transformations
        R_checkerboard_T_camera, t_checkerboard_T_camera, L_failed_id = self._target_T_camera()

        # Read base to TCP (Tool Center Point) transformation data
        file_path = os.path.join(self.dir, "base_T_tcp.json")
        data = readJSON(file_path)

        # Remove the transformations corresponding to failed checkerboard detections
        t_base_T_tcp = data.get('pos')
        R_base_T_tcp = data.get('rotvec')

        R_base_T_tcp = np.delete(R_base_T_tcp, L_failed_id, 0)
        t_base_T_tcp = np.delete(t_base_T_tcp, L_failed_id, 0)

        # Perform hand-eye calibration
        rotmat, pos = cv2.calibrateHandEye(
            R_base_T_tcp,
            t_base_T_tcp,
            R_checkerboard_T_camera,
            t_checkerboard_T_camera,
            method=method,
        )

        # Save the estimated transformation matrix
        H = composeH(rotmat, pos)

        if not os.path.exists(self.save_dir):
            os.mkdir('self.save_dir')

        if not fn.endswith(".json"):
            raise Exception("File name does not end with '.json'!")
        path = os.path.join(self.save_dir, fn)
        data = {'extrinsic': H.tolist()}

        print("-------------------")
        print("Extrinsic Estimation:")
        print(H)
        print("-------------------")
        writeJSON(data, path)

        return H
