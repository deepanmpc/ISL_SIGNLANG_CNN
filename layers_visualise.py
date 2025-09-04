import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import SignLanguageCNN

# Load the model
model = SignLanguageCNN(num_classes=35)
model.load_state_dict(torch.load('isl_sign_model.pt', map_location=torch.device('cpu')))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load and transform the image
try:
    img = Image.open("test.png").convert('RGB')
except FileNotFoundError:
    print("Error: test.png not found. Please provide a test image in the root directory.")
    exit()

img_tensor = transform(img).unsqueeze(0)

# Layers to visualise
# For a quick test, visualizing only the first layer and 8 features.
layers_to_visualise = [0] 
max_features = 8
x = img_tensor

with torch.no_grad():
    for idx, layer in enumerate(model.conc_layers):
        x = layer(x)
        if idx in layers_to_visualise:
            # Check if it's a convolutional layer with features to visualize
            if len(x.shape) == 4:
                num_features = min(x.shape[1], max_features)
                plt.figure(figsize=(15, 8))
                for i in range(num_features):
                    plt.subplot(max_features // 8, 8, i + 1)
                    plt.imshow(x[0, i].cpu().numpy(), cmap='viridis')
                    plt.axis('off')
                plt.suptitle(f'Layer {idx} Activations ({layer[0].__class__.__name__})')
                plt.show()