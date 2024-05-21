import cv2
import dlib
import numpy as np

# Load the pre-trained models for face and landmark detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Process each face found
    for face in faces:
        landmarks = predictor(gray, face)
        if not landmarks:
            continue

        # Extract coordinates for the left and right eye
        left_eye_points = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)
        ]
        right_eye_points = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)
        ]

        # Convert points to numpy arrays
        left_eye_np = np.array(left_eye_points, dtype=np.int32)
        right_eye_np = np.array(right_eye_points, dtype=np.int32)

        # Create bounding boxes around the eyes
        (x, y, w, h) = cv2.boundingRect(left_eye_np)
        left_eye_region = gray[y : y + h, x : x + w]
        (x, y, w, h) = cv2.boundingRect(right_eye_np)
        right_eye_region = gray[y : y + h, x : x + w]

        # Apply pupil tracking (ellipse fitting) within the left eye region
        blurred_left_eye = cv2.GaussianBlur(left_eye_region, (5, 5), 0)
        edges_left_eye = cv2.Canny(blurred_left_eye, 50, 150)
        contours, _ = cv2.findContours(
            edges_left_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                ellipse_center = (
                    int(ellipse[0][0] + x),
                    int(ellipse[0][1] + y),
                )  # Adjust position
                cv2.ellipse(
                    frame, (ellipse_center, ellipse[1], ellipse[2]), (0, 255, 0), 2
                )

        # Apply pupil tracking (ellipse fitting) within the right eye region
        blurred_right_eye = cv2.GaussianBlur(right_eye_region, (5, 5), 0)
        edges_right_eye = cv2.Canny(blurred_right_eye, 50, 150)
        contours, _ = cv2.findContours(
            edges_right_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                ellipse_center = (
                    int(ellipse[0][0] + x),
                    int(ellipse[0][1] + y),
                )  # Adjust position
                cv2.ellipse(
                    frame, (ellipse_center, ellipse[1], ellipse[2]), (0, 255, 0), 2
                )

    # Display the result
    cv2.imshow("Eye Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()




Network 
-model_name: String
-model
-device: String
--
+__init__(num_classes, model_name)
+set_model()
+get_model_name()
+forward(x)
+get_current_device()

Transforms 
+__init__()
-_rotate(image, landmarks, angle)
-_resize(image, landmarks, img_size)
-_color_jitter(image, landmarks)
-_crop_face(image, landmarks, crops)
+__call__(image, landmarks, crops)

Trainer
-model
-dataset
-criterion
-optimizer
-num_epochs : int
--
+__init__(model, dataset, criterion, optimizer, num_epochs)
-_print_overwrite(step, total_step, loss, operation)
-_split_data(ratio)
+train()

FaceLandmarksDataset
-transforms
--
+__init__(transform)
+__len__()
+__getitem__(index)



