from imagededup.methods import PHash, CNN

cnn_encoder = CNN()

# Generate encodings for all images in an image directory
duplicates = cnn_encoder.find_duplicates(
    image_dir="../input/siic-isic-224x224-images/train/",
    scores=True,
    outfile="duplicates.json",
    min_similarity_threshold=0.85,
)

print(duplicates)
