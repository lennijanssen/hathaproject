from operator import truediv
import numpy as np
import math as mt


# Function for getting angles for single image input,
# returns dict with angle name as key and list with only the values
# Please use 2D-values for accurate results (x,y)

def angle_function(lm):

    landmark_dict = {
    'landmarks_left_elbow' : (lm[9], lm[7], lm[5]),
    'landmarks_right_elbow' : (lm[10], lm[8], lm[6]),
    'landmarks_left_shoulder' : (lm[11], lm[5], lm[7]),
    'landmarks_right_shoulder' : (lm[12], lm[6], lm[8]),
    'landmarks_hip_left' : (lm[13], lm[11], lm[5]),
    'landmarks_hip_right' : (lm[14], lm[12], lm[6]),
    'landmarks_left_knee' : (lm[15], lm[13], lm[11]),
    'landmarks_right_knee' : (lm[16], lm[14], lm[12])
        }

    def angle_calculator(landmarks):

        # Converting the points into numpy arrays
        p1 = np.array(landmarks[0])
        p2 = np.array(landmarks[1])
        p3 = np.array(landmarks[2])

        # Creating the Vectors between two points
        vec_p1 = p1-p2
        vec_p2 = p3-p2

        # Calculating the cosine of the angle
        cosine_angle = np.dot(vec_p1,vec_p2) / (np.linalg.norm(vec_p1)*np.linalg.norm(vec_p2))

        #clipping cosine angle
        cosine_angle = np.clip(cosine_angle, -1, 1)
        # Calculating angle
        angle = np.arccos(cosine_angle)

        # Calculating degrees
        angle_degrees = np.degrees(angle)

        return angle_degrees

    angle_name_list = ['angle_elbow_left',
                   'angle_elbow_right',
                   'angle_shoulder_left',
                   'angle_shoulder_right',
                   'angle_hip_left',
                   'angle_hip_right',
                   'angle_knee_left',
                   'angle_knee_right',
                  ]

    angle_dict = {}

    for dictkey, listkey in zip(landmark_dict, angle_name_list):
        i = angle_calculator(landmark_dict[dictkey])
        angle_dict[listkey] = i

    angle_list = [angle_dict[key] for key in angle_dict]

    return angle_dict, angle_list


# This function compares two given landmark sets on their angles
# Please use 2D-values for accurate results (x,y)


def angle_comparer(test, best):

    # get angles for each pose
    best_pose_angles = angle_function(best)[1]
    test_pose_angles = angle_function(test)[1]

    # find differences between each by substraction
    angle_substraction = [a - b for a, b in zip(best_pose_angles, test_pose_angles)]

    # change values to absolute values
    angle_difference = list(map(abs, angle_substraction))
    test_angle_percentage_diff = [x/180 for x in angle_difference]

    # find percentages for each pose
    test_angle_percentage = list(map(truediv, test_pose_angles, best_pose_angles))

    # find absolute difference by percentage
    test_angle_percentage_diff_unscaled = [(abs(round((x-1),2))) for x in test_angle_percentage]
    # test_angle_percentage_diff = [abs(round((x-1),2)) for x in test_angle_percentage]
    score_angles_unscaled = [(1 - abs(round((x-1),2))) for x in test_angle_percentage]
    score_angles = [(1 - abs(round((x-1),2)))*5 for x in test_angle_percentage]

    print(test_angle_percentage_diff)
    # find average percentage difference
    average_percentage_diff = abs(round(sum(test_angle_percentage_diff_unscaled)/len(test_angle_percentage_diff_unscaled),2))
    average_score = 1-abs(round(sum(test_angle_percentage_diff_unscaled)/len(test_angle_percentage_diff_unscaled),2))
    print(average_percentage_diff)

    return test_angle_percentage_diff, average_percentage_diff, score_angles, score_angles_unscaled, average_score
