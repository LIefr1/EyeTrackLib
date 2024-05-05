import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import torchvision.transforms.functional as TF
from tutorial.model import Network
from PIL import Image
import sys
from tutorial.train import Trainer
from tutorial.Landmark_dataset import FaceLandmarksDataset
from tutorial.transforms import Transforms
import torch.optim as optim

if __name__ == "__main__":
    model = Network(model_name="resnet18")
    dataset = FaceLandmarksDataset(Transforms())
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    trainer = Trainer(model=model, dataset=dataset, optimizer=optimizer, num_epochs=2)
    trainer.train()
    # face_cascade = cv2.CascadeClassifier(
    #     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    # )

    # cap = cv2.VideoCapture(0)
    # cap.set(propId=cv2.CAP_PROP_FRAME_WIDTH, value=1920)
    # cap.set(propId=cv2.CAP_PROP_FRAME_HEIGHT, value=1080)

    # # best_network = Network(model_name="resnet50")
    # # best_network.load_state_dict(torch.load("models/face_landmarks_resnet50.pth"))
    # best_network = Network(model_name="resnet18")
    # best_network.load_state_dict(torch.load("models/face_landmarks.pth"))
    # best_network.eval()

    # while True:
    #     ret, frame = cap.read()
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     height, width, _ = frame.shape
    #     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #     all_landmarks = []
    #     for x, y, w, h in faces:
    #         image = gray[y : y + h, x : x + w]
    #         image = TF.resize(Image.fromarray(image), size=(224, 224))
    #         image = TF.to_tensor(image)
    #         image = TF.normalize(image, [0.5], [0.5])

    #         with torch.no_grad():
    #             landmarks = best_network(image.unsqueeze(0))

    #         landmarks = (landmarks.view(68, 2).cpu().detach().numpy() + 0.5) * np.array(
    #             [[w, h]]
    #         ) + np.array([[x, y]])
    #         all_landmarks.append(landmarks)
    #     # sys.exit()
    #     # sys.exit()
    #     for x, y in all_landmarks[0]:
    #         cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    #     cv2.imshow("Output", frame)
    #     k = cv2.waitKey(5) & 0xFF
    #     if k == 27:
    #         break
    # cv2.destroyAllWindows()
    # cap.release()
