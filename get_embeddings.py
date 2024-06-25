import pandas as pd
from img2vec_pytorch import Img2Vec
import image_utils

paths = image_utils.get_images_from_dir("processed_images/cat")
images = [image_utils.load_image(path) for path in paths]

# Initialize Img2Vec with GPU
img2vec = Img2Vec(cuda=True)
embeddings = img2vec.get_vec(images)

print(embeddings.shape)

df = pd.DataFrame(embeddings)
df["filepaths"] = paths
df.to_csv("embeddings/cat_embeddings.csv", index=False)
