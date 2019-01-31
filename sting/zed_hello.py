########################################################################
#
# Copyright (c) 2017, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
import math
import cv2
import time
import numpy as np
import sys

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
    init_params.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.UNIT_MILLIMETER  # Use milliliter units (for depth measurements)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD  # Use STANDARD sensing mode

    # Capture 50 images and depth, then stop
    image = sl.Mat()
    depth_map = sl.Mat()
    point_cloud = sl.Mat()
    res = zed.get_resolution()
    depth_image_zed = sl.Mat(res.width, res.height, sl.MAT_TYPE.MAT_TYPE_8U_C4)
    key = ' '
    while key != 113 :
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image, sl.VIEW.VIEW_LEFT, sl.MEM.MEM_CPU, int(res.width), int(res.height))
            zed.retrieve_image(depth_image_zed, sl.VIEW.VIEW_DEPTH, sl.MEM.MEM_CPU, int(res.width), int(res.height))
            # Retrieve the RGBA point cloud in half resolution
            zed.retrieve_measure(point_cloud, sl.MEASURE.MEASURE_XYZRGBA, sl.MEM.MEM_CPU, int(res.width), int(res.height))

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image.get_data()
            depth_image_ocv = point_cloud.get_data()

            # cv2.imshow("Image", image_ocv)
            cv2.imshow("Depth", depth_image_ocv)

            key = cv2.waitKey(10)

    cv2.destroyAllWindows()
    zed.close()


if __name__ == "__main__":
    main()
