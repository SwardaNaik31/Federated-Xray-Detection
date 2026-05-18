import os
import shutil
import random

# Original dataset paths
NORMAL_PATH = "COVID-19_Radiography_Dataset/NORMAL/images"
PNEUMONIA_PATH = "COVID-19_Radiography_Dataset/Viral Pneumonia/images"

# Hospital folders
hospitals = [
    "data/hospital_A",
    "data/hospital_B",
    "data/hospital_C"
]

# Create hospital directories
for hospital in hospitals:

    os.makedirs(f"{hospital}/NORMAL", exist_ok=True)
    os.makedirs(f"{hospital}/PNEUMONIA", exist_ok=True)

# Get all image filenames
normal_images = os.listdir(NORMAL_PATH)
pneumonia_images = os.listdir(PNEUMONIA_PATH)

# Shuffle images
random.shuffle(normal_images)
random.shuffle(pneumonia_images)

# Split function
def split_images(images, disease_name):

    total = len(images)

    split1 = int(total * 0.33)
    split2 = int(total * 0.66)

    hospital_A = images[:split1]
    hospital_B = images[split1:split2]
    hospital_C = images[split2:]

    splits = [
        hospital_A,
        hospital_B,
        hospital_C
    ]

    for i, hospital_images in enumerate(splits):

        for image in hospital_images:

            src = (
                NORMAL_PATH if disease_name == "NORMAL"
                else PNEUMONIA_PATH
            )

            src_path = os.path.join(src, image)

            dst_path = os.path.join(
                hospitals[i],
                disease_name,
                image
            )

            shutil.copy(src_path, dst_path)
import os
import shutil
import random

# Dataset paths
NORMAL_PATH = "COVID-19_Radiography_Dataset/NORMAL/images"
PNEUMONIA_PATH = "COVID-19_Radiography_Dataset/Viral Pneumonia/images"

# Hospital folders
hospitals = [
    "data/hospital_A",
    "data/hospital_B",
    "data/hospital_C"
]

# Create folders
for hospital in hospitals:

    os.makedirs(f"{hospital}/NORMAL", exist_ok=True)
    os.makedirs(f"{hospital}/PNEUMONIA", exist_ok=True)

# Get image names
normal_images = os.listdir(NORMAL_PATH)
pneumonia_images = os.listdir(PNEUMONIA_PATH)

# Shuffle
random.shuffle(normal_images)
random.shuffle(pneumonia_images)

# Small dataset for testing
normal_images = normal_images[:60]
pneumonia_images = pneumonia_images[:60]

# Split NORMAL images
split_images(normal_images, "NORMAL")

# Split PNEUMONIA images
split_images(pneumonia_images, "PNEUMONIA")

print("Dataset split successfully!")