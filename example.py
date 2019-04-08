import numpy as np
import math
import cv2
import csv

def glasses_filter(face, glasses, landmarks, pose, should_show_bounds=False):
    # Modified from "https://github.com/oflynned/Snapchat-Filter/blob/master/main.py"
    orgn_per_pts = np.array([[0, 0], [639, 0], [0, 214], [639, 214]], np.float32)
    anchor_pt = [landmarks[27][0], landmarks[27][1]]

    if type(landmarks) is not int:
        """
        GLASSES ANCHOR POINTS:
        17 & 26 edges of left eye and right eye (left and right extrema)
        0 & 16 edges of face across eyes (other left and right extra, interpolate between 0 & 17, 16 & 26 for half way points)
        19 & 24 top of left and right brows (top extreme)
        27 is centre of the eyes on the nose (centre of glasses)
        28 is the bottom threshold of glasses (perhaps interpolate between 27 & 28 if too low) (bottom extreme)
        """

        per_tl = [landmarks[17][0] - 10 + 10 * math.sin(pose[2]),
                  landmarks[17][1] - 10 * math.cos(pose[2])]
        per_tr = [landmarks[26][0] + 10 + 10 * math.sin(pose[2]),
                  landmarks[26][1] - 10 * math.cos(pose[2])]
        per_3 = [landmarks[17][0] - 10 - 30 * math.sin(pose[2]),
                  landmarks[17][1] + 30 * math.cos(pose[2])]
        per_4 = [landmarks[26][0]  + 10 - 30 * math.sin(pose[2]),
                  landmarks[26][1] + 30 * math.cos(pose[2])]

        dest_per_pts = np.array([per_tl, per_tr, per_3, per_4], np.float32)

        M = cv2.getPerspectiveTransform(orgn_per_pts, dest_per_pts)

        rotated = cv2.warpPerspective(glasses, M, (face.shape[1], face.shape[0]))

        result_2 = overlay(face, rotated)

        if should_show_bounds:
            for p in dest_per_pts:
                pos = (p[0], p[1])
                cv2.circle(result_2, pos, 2, (0, 0, 255), 2)
                cv2.putText(result_2, str(p), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))

        cv2.imshow("Glasses Filter", result_2)
        cv2.waitKey(10)
        pass


def overlay(face_img, overlay_img):
    # Modified from "https://github.com/oflynned/Snapchat-Filter/blob/master/main.py"

    # Mask RGB info
    overlay_rgb = overlay_img[:, :, :3]
    # Opacity value
    overlay_mask = overlay_img[:, :, 3:]
    # Background
    bkgd_mask = 255 - overlay_mask
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    bkgd_mask = cv2.cvtColor(bkgd_mask, cv2.COLOR_GRAY2BGR)

    other_part = (face_img * (1 / 255.0)) * (bkgd_mask * (1 / 255.0))
    overlay_part = (overlay_rgb * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # cast to 8 bit matrix
    final_img  = np.uint8(cv2.addWeighted(other_part, 255.0, overlay_part, 255.0, 0.0))

    return final_img


def readin(file_name):
    with open(file_name,mode='r') as csv_file:
        ldmks = []  # [[(x0, y0),(x1, y1),..., (x67, y67)], ...]
        poses = []  # [[Rx0, Ry0, Rz0], ...]
        csv_reader = csv.DictReader(csv_file)
        for line_cnt, row in enumerate(csv_reader):
            if line_cnt == 0:
                print(f'Column names are {", ".join(row)}')

            ldmk = []
            for i in range(68):
                ldmk.append([float(row[' x_%d' % i]), float(row[' y_%d' % i])])
            ldmks.append(ldmk)
            poses.append([float(row[' pose_Rx']), float(row[' pose_Ry']), float(row[' pose_Rz'])])
    return ldmks, poses


def main():
    ldmks, poses = readin('speech.csv')
    glasses = cv2.imread('glasses.png', -1)
    #cv2.imshow('glass', glasses)
    #cv2.waitKey(30)
    cap = cv2.VideoCapture('speech.mp4')
    frame_cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret ==True:
            ldmk = ldmks[frame_cnt]
            pose = poses[frame_cnt]

            glasses_filter(frame, ldmks[frame_cnt], glasses, poses[frame_cnt], should_show_bounds=True)
        frame_cnt += 1

if __name__ == "__main__":
    main()