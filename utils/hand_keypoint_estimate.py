# ======= this Python script is used to extract the square region enclosing the hand keypoints from an image =======
# ======= a human's hand must be included in the image =======

import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

IMAGE_PATH = './example_images/hand_example.JPG'
FILE_NAME = IMAGE_PATH.split('/')[-1].replace('.JPG', '')
SAVE_FOLDER = f"{FILE_NAME}_results"

# Load an RGB image
I = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)

# check if the directory for saving results exists
if not os.path.exists(os.path.join('./results', SAVE_FOLDER)):
    os.makedirs(os.path.join('./results', SAVE_FOLDER))

# Make a copy of the original image to use for cropping the square region
I_original = I.copy()

# Convert the BGR image to RGB
image_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

# Process the image and detect hands
results = hands.process(image_rgb)

# Draw the hand key points and skeleton
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Draw skeleton connections in red
        mp_drawing.draw_landmarks(
            I, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=8, circle_radius=20),  # Skeleton in red
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=8, circle_radius=20)   # Keypoints in red
        )

# Display the image with hand keypoints and skeleton
cv2.imshow('Hand Keypoints', I)
cv2.imwrite(os.path.join('./results', SAVE_FOLDER, "keypoints.png"), I)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Check if hand landmarks are detected
if results.multi_hand_landmarks:
    # Define indices for the required six keypoints
    indices = [0, 1, 5, 9, 13, 17]
    landmarks = []
    
    # Get the coordinates of the specified keypoints
    for hand_landmarks in results.multi_hand_landmarks:
        h, w, _ = I.shape
        for i in indices:
            lm = hand_landmarks.landmark[i]
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks.append([cx, cy])
            
    # Convert landmarks to a NumPy array
    landmarks = np.array(landmarks)

    # Calculate the minimum area rectangle for these points
    rect = cv2.minAreaRect(landmarks)

    # Get the width and height of the rectangle
    width, height = rect[1]

    # Determine the side length of the square as the shorter of width or height
    side_length = min(width, height)

    # Create a square from the side of the rectangle closer to points 5, 9, 13, 17
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Identify the start corner for the square
    # Ensure that the square is aligned along the `5-9-13-17` side
    if width > height:
        # Rotate the square to align with the short side (width side) of the rectangle
        square_pts = [
            box[0],
            box[1],
            box[1] + (box[2] - box[1]) * (side_length / np.linalg.norm(box[2] - box[1])),
            box[0] + (box[3] - box[0]) * (side_length / np.linalg.norm(box[3] - box[0]))
        ]
    else:
        # Rotate the square to align with the short side (height side) of the rectangle
        square_pts = [
            box[1],
            box[2],
            box[2] + (box[3] - box[2]) * (side_length / np.linalg.norm(box[3] - box[2])),
            box[1] + (box[0] - box[1]) * (side_length / np.linalg.norm(box[0] - box[1]))
        ]

    # Convert points to integer coordinates
    square_pts = np.int0(square_pts)

    # Draw the square on the image
    cv2.drawContours(image=I, contours=[square_pts], contourIdx=0, color=(0, 0, 0), thickness=8)

    # Extract the square region from the original image
    # Define source points and destination points for perspective transform
    src_pts = np.float32(square_pts)
    dst_pts = np.float32([[0, 0], [side_length, 0], [side_length, side_length], [0, side_length]])

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply perspective transform to get the cropped square region from the original image
    square_img = cv2.warpPerspective(I_original, M, (int(side_length), int(side_length)))

    # Display the image with the minimum enclosing square
    cv2.imshow("Hand Keypoints with Square", I)
    cv2.imwrite(os.path.join('./results', SAVE_FOLDER, "keypoints_with_square.png"), I)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display and save the extracted square region
    cv2.imshow("Extracted Square Region", square_img)
    cv2.imwrite(os.path.join('./results', SAVE_FOLDER, "extracted_region.png"), square_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()