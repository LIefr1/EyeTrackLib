import cv2
import numpy as np
import torch
import time
import torchvision.transforms.functional as TF
from landmark_detector.model import LandmarkModel
from PIL import Image
import sys
from landmark_detector.dataset import Dataset
from landmark_detector.transforms import Transforms
import torch.optim as optim
import ctypes as ct


def move_mouse_ct(x: int, y: int):
    print("moving mouse to: ", x, y)
    ct.windll.user32.SetCursorPos(x, y)
    # click_if_stable_params(x=x, y=y)


def run():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    # cap.set(propId=cv2.CAP_PROP_FRAME_WIDTH, value=1920)
    # cap.set(propId=cv2.CAP_PROP_FRAME_HEIGHT, value=1080)

    # best_network = Network(model_name="resnet50")
    # best_network.load_state_dict(torch.load("models/face_landmarks_resnet50.pth"))
    best_network = LandmarkModel(model_name="resnet152")
    best_network.load_state_dict(torch.load("models/resnet152_eye_only.pth"))

    best_network.eval()

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        all_landmarks = []
        for x, y, w, h in faces:
            image = gray[y : y + h, x : x + w]
            image = TF.resize(Image.fromarray(image), size=(224, 224))
            image = TF.to_tensor(image)
            image = TF.normalize(image, [0.5], [0.5])

            with torch.no_grad():
                landmarks = best_network(image.unsqueeze(0))

            landmarks = (landmarks.view(68, 2).cpu().detach().numpy() + 0.5) * np.array(
                [[w, h]]
            ) + np.array([[x, y]])
            all_landmarks.append(landmarks)
            # sys.exit()
            # sys.exit()
            for landmarks in all_landmarks:
                left_eye = [
                    landmarks[36],
                    landmarks[37],
                    landmarks[38],
                    landmarks[39],
                    landmarks[40],
                    landmarks[41],
                ]
                right_eye = [
                    landmarks[42],
                    landmarks[43],
                    landmarks[44],
                    landmarks[45],
                    landmarks[46],
                    landmarks[47],
                ]

                left_eye_x = [point[0] for point in left_eye]
                left_eye_y = [point[1] for point in left_eye]

                right_eye_x = [point[0] for point in right_eye]
                right_eye_y = [point[1] for point in right_eye]

                # Combine for easier calculation (optional)
                all_eye_x = left_eye_x + right_eye_x
                all_eye_y = left_eye_y + right_eye_y

                # Find bounding box coordinates
                left_eye_min_x = int(min(left_eye_x))
                left_eye_max_x = int(max(left_eye_x))
                left_eye_min_y = int(min(left_eye_y))
                left_eye_max_y = int(max(left_eye_y))

                right_eye_min_x = int(min(right_eye_x))
                right_eye_max_x = int(max(right_eye_x))
                right_eye_min_y = int(min(right_eye_y))
                right_eye_max_y = int(max(right_eye_y))
                # move_mouse_ct(int(landmarks[36][0]), int(landmarks[36][0]))
                # cv2.rectangle(
                #     frame,
                #     (left_eye_min_x, left_eye_min_y),
                #     (left_eye_max_x, left_eye_max_y),
                #     (0, 255, 0),
                #     2,
                # )
                # cv2.rectangle(
                #     frame,
                #     (right_eye_min_x, right_eye_min_y),
                #     (right_eye_max_x, right_eye_max_y),
                #     (0, 255, 0),
                #     2,
                # )
                for x, y in landmarks:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        # x = all_landmarks[0][36][0]
        # y = all_landmarks[0][36][1]
        # print(x, y)
        # cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.imshow("Output", frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    # model = Network(
    #     model_name="resnet18",
    # )
    #  Dataset(Transforms())
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # trainer = Trainer(model=model, dataset=dataset, optimizer=optimizer, num_epochs=2)
    # trainer.train()
    run()
