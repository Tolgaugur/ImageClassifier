import pandas as pd
from img2vec_pytorch import Img2Vec
import image_utils

paths = image_utils.get_images_from_dir("./processed_images/dog")
images = [image_utils.load_image(path) for path in paths]


img2Vec = Img2Vec(cuda=True)

embeddings = img2Vec.get_vec(images)

print(embeddings.shape)


df = pd.DataFrame(embeddings)
df["filepaths"] = paths

df.to_csv("./embeddings/dog_embeddings.csv", index=False)
