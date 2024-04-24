import torch
import cv2
from PIL import Image  # For image conversion
import torchvision.transforms as transforms
from torchvision.models import resnet18


def preprocess_image(image):
    """
    Preprocess the image for the model.

    Args:
        image: OpenCV BGR image frame.

    Returns:
        A PyTorch tensor representing the preprocessed image.
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to PIL image
    pil_image = Image.fromarray(rgb_image)
    # Create a PyTorch tensor with desired transformations (adjust as needed) and return it
    return transforms.Compose(
        [
            transforms.Resize(256),  # Resize to a specific size
            transforms.CenterCrop(224),  # Center crop to a specific size
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize based on model requirements
        ]
    )(pil_image)


model = resnet18(num_classes=4)


model.load_state_dict(torch.load("B:/EyeTrackLib/models/model_conv18_3.pt"))
model.eval()  # Set the model to evaluation mode

cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Error: Frame not captured!")
        break

    # Preprocess the captured frame
    preprocessed_image = preprocess_image(frame)

    # Add a batch dimension for the model (assuming it expects batches)
    preprocessed_image = preprocessed_image.unsqueeze(0)

    # Move the tensor to the device (CPU or GPU)
    # preprocessed_image = preprocessed_image.to(model.device)

    # Perform inference
    with torch.no_grad():
        outputs = model(preprocessed_image)

    # Get the predicted class (assuming model outputs class probabilities)
    _, predicted_class = torch.max(outputs.data, 1)
    predicted_class = predicted_class.item()

    # Display the frame and predicted class
    cv2.putText(
        frame,
        f"Predicted Class: {predicted_class}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Webcam Feed with Prediction", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
