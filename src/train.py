import numpy as np
import cv2 as cv
import sys

# Initialize video capture
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 30)
cap.set(propId=cv.CAP_PROP_FRAME_WIDTH, value=800)
cap.set(propId=cv.CAP_PROP_FRAME_HEIGHT, value=800)
if not cap.isOpened():
    raise Exception("Cannot open camera")

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    raise Exception("Cannot read first frame from camera")

# Convert the first frame to grayscale
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

# Load the Haar cascade for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces in the first frame
rects = face_cascade.detectMultiScale(prvs, scaleFactor=1.1, minNeighbors=4)

# Create a mask for the detected face
mask = np.zeros_like(prvs, dtype=np.uint8)
RECT_H, RECT_W = 0, 0
for x, y, w, h in rects:
    RECT_H, RECT_W = h, w
    cv.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

while True:
    # Read the next frame
    ret, frame2 = cap.read()
    if not ret:
        print("No frames grabbed!")
        break

    # Convert the new frame to grayscale
    next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method
    flow = cv.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Apply the mask to the optical flow
    flow_masked = np.zeros_like(flow)
    flow_masked[mask == 255] = flow[mask == 255]

    # Convert flow to polar coordinates (magnitude and angle)
    # mag, ang = cv.cartToPolar(flow_masked[..., 0], flow_masked[..., 1])
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Create an HSV image with the optical flow
    # Convert flow to HSV format, where H is the angle and S,V are the magnitude
    # Normalize the magnitude values to the range [0, 255]
    hsv_masked = np.zeros_like(frame2, dtype=np.uint8)
    hsv_masked[..., 1] = 255  # Saturation
    hsv_masked[..., 0] = ang * 180 / np.pi / 2  # Hue (converted to degrees)
    print("max mag", np.max(mag))
    if np.max(mag) > 15.0:
        hsv_masked[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # Vaelue (magnitude)

    # Convert HSV image to BGR format for display
    bgr_masked = cv.cvtColor(hsv_masked, cv.COLOR_HSV2BGR)

    # Extract and print coordinates of significant flow vectors
    threshold = 10.0  # Threshold for magnitude of flow vectors
    # print("mag", mag)
    y_indices, x_indices = np.where(mag > threshold)
    if x_indices.size > 0 and y_indices.size > 0:
        vectors = np.array((x_indices, y_indices))
        # print("Vectors", vectors)
        result_vectors = np.sum(vectors, axis=1) / len(vectors)
        # print("resilt_vectors", result_vectors)
        # sys.exit()
        magnitude = np.linalg.norm(result_vectors)
        if magnitude != 0:
            cumulative_direction = result_vectors / magnitude
            h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
            w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
            start_point = (int(w / 2), int(h / 2))
            # print("Start point", start_point)
            # point = cumulative_direction * np.array([[w, h]])[0]
            # print("Point", point)
            # end_point = (
            #     int(start_point[0] + cumulative_direction[0] * 100),
            #     int(start_point[1] + cumulative_direction[1] * 100),
            # )
            # cv.arrowedLine(bgr_masked, start_point, end_point, (0, 255, 0), 3)
            # cv.circle(
            #     bgr_masked,
            #     (int(point[0]), int(point[1])),
            #     5,
            #     (0, 255, 0),
            #     -1,
            # )
        # np.set_printoptions(threshold=np.inf)
        # print("Magnitude:", magnitude)
        # print("Resultant Vector:", result_vectors)
        # print("Cumulative Direction:", cumulative_direction)

    cv.imshow("Optical Flow (Head Region)", bgr_masked)

    # Exit on ESC key press
    if cv.waitKey(30) & 0xFF == 27:
        break

    # Update the previous frame
    prvs = next_frame.copy()

# Release the capture and destroy all windows
cap.release()
cv.destroyAllWindows()
