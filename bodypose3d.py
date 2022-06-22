import math

import cv2 as cv
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import DLT, get_projection_matrix, write_keypoints_to_disk

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

frame_shape = [480, 640]

#add here if you need more keypoints
#pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]
pose_keypoints = [ 0, 2, 5, 7, 8, 9, 10, 16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28 ]

plt.style.use('seaborn')

def visualize_3d(fig, ax, p3ds):
    ax.set_xlim3d(-10, 5)
    ax.set_ylim3d(-5, 10)
    ax.set_zlim3d(15, 35)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    """Now visualize in 3D"""
    # torso = [[0, 1] , [1, 7], [7, 6], [6, 0]]
    # armr = [[1, 3], [3, 5]]
    # arml = [[0, 2], [2, 4]]
    # #legr = [[6, 8], [8, 10]]
    # #legl = [[7, 9], [9, 11]]

    head = [[0, 1], [1, 3], [0, 2], [2, 4]]
    mouth = [[5, 6]]
    torso = [[7, 8] , [8, 14], [14, 13], [13, 7]]
    armr = [[8, 10], [10, 12]]
    arml = [[7, 9], [9, 11]]
    #legr = [[6+7, 8+7], [8+7, 10+7]]
    #legl = [[7+7, 9+7], [9+7, 11+7]]

    #body = [torso, arml, armr, legr, legl]
    body = [torso, arml, armr, head, mouth]
    #colors = ['red', 'blue', 'green', 'black', 'orange']
    colors = ['red', 'blue', 'green', 'orange', 'cyan']

    from mpl_toolkits.mplot3d import Axes3D

    kpts3d = p3ds

    for bodypart, part_color in zip(body, colors):
        for _c in bodypart:
            ax.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = part_color)

    # Print distance between nose and (0, 0, 0):
    nose_x = kpts3d[0][0]
    nose_y = kpts3d[0][1]
    nose_z = kpts3d[0][2]
    nose_dist = math.sqrt(nose_x**2 + nose_y**2 + nose_z**2) * 6 # multiply by 6cm - this is the square length
    print(nose_dist)

    #uncomment these if you want scatter plot of keypoints and their indices.
    # for i in range(12):
    #     #ax.text(kpts3d[i,0], kpts3d[i,1], kpts3d[i,2], str(i))
    #     #ax.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])

    plt.pause(0.04)
    ax.cla()

def run_mp(input_stream1, input_stream2, P0, P1):
    #input video stream
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    #set camera resolution if using webcam to 640x480. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    #create body keypoints detector objects.
    pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    #containers for detected keypoints for each camera. These are filled at each frame.
    #This will run you into memory issue if you run the program without stop
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_axis_off()
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])

    ax.set_xlim3d(-20, -10)
    ax.set_ylim3d(-8, -4)
    ax.set_zlim3d(-10, 0)

    #ax.set_xlim3d(-20, 20)
    ax.set_xlabel('x')
    #ax.set_ylim3d(-20, 20)
    ax.set_ylabel('y')
    #ax.set_zlim3d(-20, 20)
    ax.set_zlabel('z')

    while True:

        #read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1: break

        #crop to 480x480.
        #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        if frame0.shape[1] != 480:
            frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # the BGR image to RGB.
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = pose0.process(frame0)
        results1 = pose1.process(frame1)

        #reverse changes
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

        # print("\n")
        # print(results0.pose_landmarks)
        # print("\n")

        #check for keypoints detection
        frame0_keypoints = []
        if results0.pose_landmarks:
            for i, landmark in enumerate(results0.pose_landmarks.landmark):
                if i not in pose_keypoints: continue #only save keypoints that are indicated in pose_keypoints
                pxl_x = landmark.x * frame0.shape[1]
                pxl_y = landmark.y * frame0.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv.circle(frame0,(pxl_x, pxl_y), 3, (0,255,0), -1) #add keypoint detection points into figure
                kpts = [pxl_x, pxl_y]
                frame0_keypoints.append(kpts)
        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame0_keypoints = [[-1, -1]]*len(pose_keypoints)

        #this will keep keypoints of this frame in memory
        kpts_cam0.append(frame0_keypoints)

        frame1_keypoints = []
        if results1.pose_landmarks:
            for i, landmark in enumerate(results1.pose_landmarks.landmark):
                if i not in pose_keypoints: continue
                pxl_x = landmark.x * frame1.shape[1]
                pxl_y = landmark.y * frame1.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv.circle(frame1,(pxl_x, pxl_y), 3, (255,255,0), -1)
                kpts = [pxl_x, pxl_y]
                frame1_keypoints.append(kpts)

        else:
            #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame1_keypoints = [[-1, -1]]*len(pose_keypoints)

        #update keypoints container
        kpts_cam1.append(frame1_keypoints)

        #calculate 3d position
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2) #calculate 3d position of keypoint
            frame_p3ds.append(_p3d)

        #print("\n")
        #print(frame_p3ds)
        #print("\n")

        '''
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        '''
        frame_p3ds = np.array(frame_p3ds).reshape((len(pose_keypoints), 3))
        kpts_3d.append(frame_p3ds)

        #print("\n\n", frame_p3ds, "\n\n")
        visualize_3d(fig, ax, frame_p3ds)

        # uncomment these if you want to see the full keypoints detections
        # mp_drawing.draw_landmarks(frame0, results0.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                           landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        #
        # mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                           landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv.imshow('cam1', frame1)
        cv.imshow('cam0', frame0)

        k = cv.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.


    cv.destroyAllWindows()
    for cap in caps:
        cap.release()


    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)

if __name__ == '__main__':

    #this will load the sample videos if no camera ID is given
    input_stream1 = 'media/cam0_test.mp4'
    input_stream2 = 'media/cam1_test.mp4'

    #put camera id as command line arguements
    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    #get projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)

    #this will create keypoints file in current working folder
    write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('kpts_3d.dat', kpts_3d)
