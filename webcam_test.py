import sys
print(sys.executable)
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
from model import SignLanguageCNN

class_names = sorted(os.listdir("/Users/deepandee/Desktop/ISL_SIGNLANG_CNN/DATA/ISL_TRAINING-1"))[:35]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("isl_sign_model.pt", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame (OpenCV BGR to RGB)
    img_rgb = cv2.cvtColor(frame, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    # Show frame with prediction
    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam Sign Prediction", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
