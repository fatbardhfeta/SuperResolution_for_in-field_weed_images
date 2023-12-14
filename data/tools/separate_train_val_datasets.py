import os
import random
import shutil

def split_train_val_images(source_dir, train_dir, val_dir, val_ratio, random_seed):
    # Set the random seed
    random.seed(random_seed)


    # Get the list of image filenames in the source directory
    image_names = os.listdir(source_dir)
    num_images = len(image_names)

    # Calculate the number of validation images based on the validation ratio
    num_val_images = int(num_images * val_ratio)

    # Generate shuffled indices
    indices = list(range(num_images))
    random.shuffle(indices)

    # Split the indices into train and validation sets
    train_indices = indices[:-num_val_images]
    val_indices = indices[-num_val_images:]

    # Move validation images to the validation directory

        # Create train and validation directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for index in val_indices:
        image = image_names[index]
        src_path = os.path.join(source_dir, image)
        dst_path = os.path.join(val_dir, image)
        shutil.move(src_path, dst_path)

    # Move train images to the train directory
    for index in train_indices:
        image = image_names[index]
        src_path = os.path.join(source_dir, image)
        dst_path = os.path.join(train_dir, image)
        shutil.move(src_path, dst_path)
    print("Images successfully split into train and validation folders.")
