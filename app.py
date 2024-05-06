import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import copy
import time
import numpy as np
import itertools
import tensorflow as tf
import sys
import csv
import argparse

model2D = tf.keras.models.load_model('model/gesture_classifier2D.keras')
model3D = tf.keras.models.load_model('model/gesture_classifier3D.keras')
hand_gesture_map = 'model/hand_gesture_label.csv'
hand_gesture = {index:gesture for index, gesture in zip(np.loadtxt(hand_gesture_map, delimiter=',', dtype='int32', usecols=(0), encoding='utf-8-sig'), np.loadtxt(hand_gesture_map, delimiter=',', dtype='str', usecols=(1), encoding='utf-8-sig'))}

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def calc_landmark_list_3D(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])

    return landmark_point

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def pre_process_landmark_3D(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    max_value = 0
    for _, landmark_point in enumerate(temp_landmark_list):
        max_value = max(max_value, abs(landmark_point[0]))
        max_value = max(max_value, abs(landmark_point[1]))

    # Convert to relative coordinates
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]

        temp_landmark_list[index][0] = (temp_landmark_list[index][0] - base_x) / max_value
        temp_landmark_list[index][1] = (temp_landmark_list[index][1] - base_y) / max_value
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    return temp_landmark_list

def draw_bounding_rect(image, brect, label:str):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                    (0, 0, 0), 3)
    cv.putText(image, label, (brect[0], brect[1]-5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    return image

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2])[:2], tuple(landmark_point[3])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2])[:2], tuple(landmark_point[3])[:2],
                (255, 0, 0), 2)
        cv.line(image, tuple(landmark_point[3])[:2], tuple(landmark_point[4])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3])[:2], tuple(landmark_point[4])[:2],
                (255, 0, 0), 2)
        cv.line(image, tuple(landmark_point[1])[:2], tuple(landmark_point[2])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1])[:2], tuple(landmark_point[2])[:2],
                (255, 0, 0), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5])[:2], tuple(landmark_point[6])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5])[:2], tuple(landmark_point[6])[:2],
                (0, 255, 0), 2)
        cv.line(image, tuple(landmark_point[6])[:2], tuple(landmark_point[7])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6])[:2], tuple(landmark_point[7])[:2],
                (0, 255, 0), 2)
        cv.line(image, tuple(landmark_point[7])[:2], tuple(landmark_point[8])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7])[:2], tuple(landmark_point[8])[:2],
                (0, 255, 0), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9])[:2], tuple(landmark_point[10])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9])[:2], tuple(landmark_point[10])[:2],
                (0, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10])[:2], tuple(landmark_point[11])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10])[:2], tuple(landmark_point[11])[:2],
                (0, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11])[:2], tuple(landmark_point[12])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11])[:2], tuple(landmark_point[12])[:2],
                (0, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13])[:2], tuple(landmark_point[14])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13])[:2], tuple(landmark_point[14])[:2],
                (255, 0, 255), 2)
        cv.line(image, tuple(landmark_point[14])[:2], tuple(landmark_point[15])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14])[:2], tuple(landmark_point[15])[:2],
                (255, 0, 255), 2)
        cv.line(image, tuple(landmark_point[15])[:2], tuple(landmark_point[16])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15])[:2], tuple(landmark_point[16])[:2],
                (255, 0, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17])[:2], tuple(landmark_point[18])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17])[:2], tuple(landmark_point[18])[:2],
                (119, 173, 235), 2)
        cv.line(image, tuple(landmark_point[18])[:2], tuple(landmark_point[19])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18])[:2], tuple(landmark_point[19])[:2],
                (119, 173, 235), 2)
        cv.line(image, tuple(landmark_point[19])[:2], tuple(landmark_point[20])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19])[:2], tuple(landmark_point[20])[:2],
                (119, 173, 235), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0])[:2], tuple(landmark_point[1])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0])[:2], tuple(landmark_point[1])[:2],
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[0])[:2], tuple(landmark_point[5])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0])[:2], tuple(landmark_point[5])[:2],
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[5])[:2], tuple(landmark_point[9])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5])[:2], tuple(landmark_point[9])[:2],
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[9])[:2], tuple(landmark_point[13])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9])[:2], tuple(landmark_point[13])[:2],
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[13])[:2], tuple(landmark_point[17])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13])[:2], tuple(landmark_point[17])[:2],
                (0, 0, 255), 2)
        cv.line(image, tuple(landmark_point[17])[:2], tuple(landmark_point[0])[:2],
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17])[:2], tuple(landmark_point[0])[:2],
                (0, 0, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def inference(landmark_point, mode:int):
    if mode == 2:
        predict_result = model2D.predict(tf.expand_dims(np.array(landmark_point), 0))
    else:
        predict_result = model3D.predict(tf.expand_dims(np.array(landmark_point), 0))
    return np.argmax(predict_result, 1)[0]

def log_new_instance(label: int, landmark_list, mode:int):
    csv_path = f'model/keypoint{mode}D.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, *landmark_list])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--add", type=int, default=-1)
    parser.add_argument("--mode", type=int, default=2)
    return parser.parse_args()


def main():
    args = get_args()
    mode = args.mode
    if mode != 2 and mode != 3:
        return
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    count = 0
    while True:
        lognew = False
        key = cv.waitKey(10)
        if key == 27:
            break
        if key == ord('l'):
            lognew = True

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = []
                if mode == 3:
                    landmark_list = calc_landmark_list_3D(debug_image, hand_landmarks)
                else:
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = []
                if mode == 3:
                    pre_processed_landmark_list = pre_process_landmark_3D(landmark_list)
                else:
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                if lognew and args.add > -1:
                    print(f'Logged {count}')
                    count+=1
                    log_new_instance(args.add, pre_processed_landmark_list, mode)

                if args.add == -1:
                    label = inference(pre_processed_landmark_list, mode)

                    # Drawing part
                    debug_image = draw_bounding_rect(debug_image, brect, hand_gesture[label])
                debug_image = draw_landmarks(debug_image, landmark_list)
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()