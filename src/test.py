import numpy as np
import cv2 as cv

# Initialize video capture
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Cannot open camera")

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    raise Exception("Cannot read first frame from camera")

# Convert the first frame to grayscale
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

# Load the Haar cascade for face detection
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Detect faces in the first frame
rects = face_cascade.detectMultiScale(prvs, scaleFactor=1.1, minNeighbors=4)

# Create a mask for the detected face
mask = np.zeros_like(prvs, dtype=np.uint8)
for x, y, w, h in rects:
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
    mag, ang = cv.cartToPolar(flow_masked[..., 0], flow_masked[..., 1])

    # Create an HSV image with the optical flow
    hsv_masked = np.zeros_like(frame2, dtype=np.uint8)
    hsv_masked[..., 1] = 255
    hsv_masked[..., 0] = ang * 180 / np.pi / 2
    hsv_masked[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    # Convert HSV image to BGR format for display
    bgr_masked = cv.cvtColor(hsv_masked, cv.COLOR_HSV2BGR)

    # Extract and print coordinates of significant flow vectors
    threshold = 1.2  # Threshold for magnitude of flow vectors
    y_indices, x_indices = np.where(mag > threshold)
    coords = list(zip(x_indices, y_indices))
    print("Significant flow vector coordinates within the face region:", coords)

    # Display the resulting frame
    cv.imshow("Optical Flow (Head Region)", bgr_masked)

    # Exit on ESC key press
    if cv.waitKey(30) & 0xFF == 27:
        break

    # Update the previous frame
    prvs = next_frame.copy()

# Release the capture and destroy all windows
cap.release()
cv.destroyAllWindows()
