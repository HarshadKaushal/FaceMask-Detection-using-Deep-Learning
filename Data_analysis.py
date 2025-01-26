import os
import matplotlib.pyplot as plt
from collections import Counter

# Define the dataset directory
dataset_dir = 'datasets/train'  # Change this to your dataset path

# Count images in each class
class_counts = Counter()
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_path):
        class_counts[class_name] = len(os.listdir(class_path))

# 6. Class Distribution Bar Chart
plt.figure(figsize=(8, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.title('Class Distribution')
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.show()