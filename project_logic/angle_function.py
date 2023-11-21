import numpy as np

def angle_function(image_landmarks_array):

    lm = image_landmarks_array

    landmark_dict = {
    'landmarks_left_elbow' : (lm[5], lm[7], lm[9]),
    'landmarks_right_elbow' : (lm[6], lm[8], lm[10]),
    'landmarks_left_shoulder' : (lm[7], lm[9], lm[11]),
    'landmarks_right_shoulder' : (lm[8], lm[10], lm[12]),
    'landmarks_hip_left' : (lm[5], lm[11], lm[13]),
    'landmarks_hip_right' : (lm[6], lm[12], lm[14]),
    'landmarks_left_knee' : (lm[11], lm[13], lm[15]),
    'landmarks_right_knee' : (lm[12], lm[14], lm[16])
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
