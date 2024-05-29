import numpy as np
import cv2 as cv

# Initialize video capture
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 30)
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
    # Convert flow to HSV format, where H is the angle and S,V are the magnitude
    # Normalize the magnitude values to the range [0, 255]
    hsv_masked = np.zeros_like(frame2, dtype=np.uint8)
    hsv_masked[..., 1] = 255  # Saturation
    hsv_masked[..., 0] = ang * 180 / np.pi / 2  # Hue (converted to degrees)
    hsv_masked[..., 2] = cv.normalize(
        mag, None, 0, 255, cv.NORM_MINMAX
    )  # Value (magnitude)

    # Convert HSV image to BGR format for display
    bgr_masked = cv.cvtColor(hsv_masked, cv.COLOR_HSV2BGR)

    # Extract and print coordinates of significant flow vectors
    threshold = 1.2  # Threshold for magnitude of flow vectors
    y_indices, x_indices = np.where(mag > threshold)
    if x_indices.size > 0 and y_indices.size > 0:
        vectors = np.array((x_indices, y_indices))
        print("Vectors", vectors)
        result_vectors = np.sum(vectors, axis=0)
        magnitude = np.linalg.norm(result_vectors)
        if magnitude != 0:
            cumulative_direction = result_vectors / magnitude
            h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
            w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
            start_point = (int(w / 2), int(h / 2))
            print("Start point", start_point)
            end_point = (
                int(start_point[0] + cumulative_direction[0] * 10),
                int(start_point[1] + cumulative_direction[1] * 10),
            )
            cv.arrowedLine(bgr_masked, start_point, end_point, (0, 255, 0), 3)

        np.set_printoptions(threshold=np.inf)
        print("Magnitude:", magnitude)
        print("Resultant Vector:", result_vectors)
        print("Cumulative Direction:", cumulative_direction)

    cv.imshow("Optical Flow (Head Region)", bgr_masked)

    # Exit on ESC key press
    if cv.waitKey(30) & 0xFF == 27:
        break

    # Update the previous frame
    prvs = next_frame.copy()

# Release the capture and destroy all windows
cap.release()
cv.destroyAllWindows()