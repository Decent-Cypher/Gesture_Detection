import urllib
import urllib.request
import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IMAGE_FILENAMES = ['thumbs_down.jpg', 'victory.jpg', 'thumbs_up.jpg', 'pointing_up.jpg']

# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{name}'
#   urllib.request.urlretrieve(url, name)

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow("None", img)
  
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

images = []
results = []
for image_file_name in IMAGE_FILENAMES:
    # STEP 3: Load the input image.
    frame = mp.Image.create_from_file(image_file_name)

    # STEP 4: Recognize gestures in the input image.
    recognition_result = recognizer.recognize(frame)

    # STEP 5: Process the result. In this case, visualize it.
    images.append(frame)
    top_gesture = recognition_result.gestures[0][0]
    hand_landmarks = recognition_result.hand_landmarks
    results.append((top_gesture, hand_landmarks))

# For webcam input:
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
        continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # results = hands.process(image)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    recognition_result = recognizer.recognize(image)
    try:
        top_gesture = recognition_result.gestures[0][0]
        print(top_gesture)
        hand_world_landmarks = recognition_result.hand_world_landmarks

        # Draw the hand annotations on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # mp_drawing.draw_landmarks(
        #     frame,
        #     hand_landmarks,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing_styles.get_default_hand_landmarks_style(),
        #     mp_drawing_styles.get_default_hand_connections_style())
        
        # Flip the image horizontally for a selfie-view display.
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print("No gestures detected")
    cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
