import cv2                      #opencv for webcam
import mediapipe as mp          # mediapipe for pose detection
#import numpy as np

# loading transparent pngs
hat = cv2.imread('ranger_hat.png', cv2.IMREAD_UNCHANGED)
vest = cv2.imread('ranger_vest.png', cv2.IMREAD_UNCHANGED)

#if hat is None:
#    raise FileNotFoundError("Could not load 'ranger_hat.png'. Make sure it's in the project folder.")
#if vest is None:
#    raise FileNotFoundError("Could not load 'ranger_vest.png'. Make sure it's in the project folder.")

#mediapipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

#overlaying transparent image onto background image
def overlay_transparent(background, overlay, x, y, scale=1):
    #resizes the overlay image based on scale
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape

    #ensures the overlay stays within frame boundaries
    if x + w > background.shape[1] or y + h > background.shape[0] or x < 0 or y < 0:
        return background

    #loop through every pixel of the overlay image
    for i in range(h):
        for j in range(w):
            #only copy pixels where alpha channel is not 0 (visible)
            if overlay[i, j][3] != 0:
                #copy RGB values to the corresponding pixel in the background
                background[y + i, x + j] = overlay[i, j][:3]

    return background

#open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    #read a frame from webcam
    ret, frame = cap.read()
    if not ret:
        break  #if frame couldn't be read, exit loop

    #flip the frame horizontally (mirror image for natural webcam feel)
    frame = cv2.flip(frame, 1)

    #convert the image to RGB for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #process the image to find pose landmarks
    result = pose.process(rgb)

    #if pose landmarks were detected
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        #detect the nose and both shoulders (key points)
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        #get the dimensions of the frame
        h, w, _ = frame.shape

        #convert coordinates to pixel values
        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
        left_x, left_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
        right_x, right_y = int(right_shoulder.x * w), int(right_shoulder.y * h)

        #estimate how wide the shoulders are (used to scale the hat and vest)
        shoulder_width = abs(right_x - left_x)
        if shoulder_width < 10:
            continue  #kkip frame if tracking failed

        #vest
        hat_scale = shoulder_width / hat.shape[1]
        vest_scale = (shoulder_width / vest.shape[1]) * 1.8  #change value on end by .1 at a time if needed to change size

        #overlay the hat
        hat_x = nose_x - int(hat.shape[1] * hat_scale) // 2
        hat_y = nose_y - int(0.7 * hat.shape[0] * hat_scale)  #change first value by .1 at a time if needed to change size

        frame = overlay_transparent(
            frame,
            hat,
            hat_x,
            hat_y,
            hat_scale
        )

        #overlay the vest
        vest_x = int((left_x + right_x) / 2) - int(vest.shape[1] * vest_scale / 2)
        vest_y = int((left_y + right_y) / 2) - int(0.2 * vest.shape[0] * vest_scale)

        frame = overlay_transparent(
            frame,
            vest,
            vest_x,
            vest_y,
            vest_scale
        )

    #show the frame in a window
    cv2.imshow("Ranger Filter", frame)

    #ESC to exit
    if cv2.waitKey(5) & 0xFF == 27:
        break

#release the webcam and close all opencv windows
cap.release()
cv2.destroyAllWindows()