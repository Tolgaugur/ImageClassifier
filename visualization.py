import matplotlib.pyplot as plt

from PIL import Image
from train import transform


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")

    return image, transform(image).unsqueeze(0)


def visualize_prediction(original_image, probablities, class_names):
    fig, axrr = plt.subplots(1, 2, figsize=(10, 5))

    axrr[0].imshow(original_image)
    axrr[0].axis("off")

    axrr[1].barh(class_names, probablities)
    axrr[1].set_xlabel("Probablity")
    axrr[1].set_title("Predictions")
    axrr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()
