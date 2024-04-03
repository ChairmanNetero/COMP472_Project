import sys
import torch
import torchvision.transforms as transforms
from Architectures import Main
from PIL import Image


# Define transform for test images
transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load the image
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print("Image not found.")
        sys.exit(1)

    # Preprocess the image
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Load the saved model
    model_path = 'models/best_model_Main.model'
    state_dict = torch.load(model_path)

    model = Main(num_classes=4)

    # Load the model's state dictionary
    model.load_state_dict(state_dict)

    model.eval()

    # Perform inference on the image
    with torch.no_grad():
        output = model(input_batch)

    # Process the output
    # For example, if you have a classification model, you might want to get the predicted class
    predicted_class = torch.argmax(output, dim=1).item()

    if predicted_class == 0:
        prediction = "Engaged"
    elif predicted_class == 1:
        prediction = "Happy"
    elif predicted_class == 2:
        prediction = "Neutral"
    elif predicted_class == 3:
        prediction = "Surprised"
    else:
        prediction = "Error"

    print("Predicted class:", prediction)
