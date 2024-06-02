import numpy as np
import cv2 as cv
import ctypes as ct


def move_mouse_ct(x: int, y: int):
    print("moving mouse to: ", x, y)
    ct.windll.user32.SetCursorPos(x, y)
    # clic


# Initialize video capture
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 30)
cap.set(propId=cv.CAP_PROP_FRAME_WIDTH, value=640)
cap.set(propId=cv.CAP_PROP_FRAME_HEIGHT, value=640)
if not cap.isOpened():
    raise Exception("Cannot open camera")

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    raise Exception("Cannot read first frame from camera")

# Convert the first frame to grayscale
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

# Load the Haar cascades for face and eye detection
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

while True:
    # Read the next frame
    ret, frame2 = cap.read()
    if not ret:
        print("No frames grabbed!")
        break

    # Convert the new frame to grayscale
    next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    rects = face_cascade.detectMultiScale(next_frame, 1.1, 4)

    # Create a mask for the detected face
    mask = np.zeros_like(next_frame, dtype=np.uint8)
    RECT_H, RECT_W = 0, 0
    for x, y, w, h in rects:
        RECT_H, RECT_W = h, w
        cv.rectangle(mask, (x, y), (x + w, y + h), 0, -1)

        # Detect eyes in the face region
        roi_gray = next_frame[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for ex, ey, ew, eh in eyes:
            eye_mask = np.zeros_like(roi_gray, dtype=np.uint8)
            cv.rectangle(eye_mask, (ex, ey), (ex + ew, ey + eh), 255, -1)
            mask[y : y + h, x : x + w] += eye_mask

    # Calculate optical flow using Farneback method
    flow = cv.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Apply the mask to the optical flow
    flow_masked = np.zeros_like(flow)
    flow_masked[mask == 255] = flow[mask == 255]

    # Convert flow to polar coordinates (magnitude and angle)
    mag, ang = cv.cartToPolar(flow_masked[..., 0], flow_masked[..., 1])

    # Create an HSV image with the optical flow
    hsv_masked = np.zeros_like(frame2, dtype=np.uint8)
    hsv_masked[..., 1] = 255  # Saturation
    hsv_masked[..., 0] = ang * 180 / np.pi / 2  # Hue (converted to degrees)
    threshold = 1.0
    if np.max(mag) > threshold:
        print("current mag", np.max(mag))
        hsv_masked[..., 2] = cv.normalize(
            mag, None, 0, 255, cv.NORM_MINMAX
        )  # Vaelue (magnitude)

    bgr_masked = cv.cvtColor(hsv_masked, cv.COLOR_HSV2BGR)
    y_indices, x_indices = np.where(mag > threshold)
    if x_indices.size > 0 and y_indices.size > 0:
        point = [np.max(x_indices), np.max(y_indices)]
        move_mouse_ct(
            int(point[0]),
            int(point[1]),
        )
        # print("x_indices", x_indices)
        # print("y_indices", y_indices)
        # vectors = np.array((x_indices, y_indices))
        # # print("Vectors", vectors)
        # result_vectors = np.sum(vectors, axis=1) / len(vectors)
        # # print("resilt_vectors", result_vectors)
        # # sys.exit()
        # magnitude = np.linalg.norm(result_vectors)
        # if magnitude != 0:
        #     cumulative_direction = result_vectors / magnitude
        #     h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        #     w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        #     start_point = (int(w / 2), int(h / 2))
        #     # print("Start point", start_point)
        #     point = cumulative_direction * np.array([[w, h]])[0]
        #     print("Point", point)
        #     # end_point = (
        #     int(start_point[0] + cumulative_direction[0] * 100),
        #     int(start_point[1] + cumulative_direction[1] * 100),
        # )
        # cv.arrowedLine(bgr_masked, start_point, end_point, (0, 255, 0), 3)
        cv.circle(
            bgr_masked,
            (int(point[0]), int(point[1])),
            5,
            (0, 255, 0),
            -1,
        )

    cv.imshow("Optical Flow (Head Region)", bgr_masked)

    # Exit on ESC key press
    if cv.waitKey(30) & 0xFF == 27:
        break

    # Update the previous frame
    prvs = next_frame.copy()

# Release the capture and destroy all windows
cap.release()
cv.destroyAllWindows()
