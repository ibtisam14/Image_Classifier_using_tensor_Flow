import kagglehub

print("Downloading dataset...")
path = kagglehub.dataset_download("sumn2u/garbage-classification-v2")
print("Dataset downloaded to:", path)
