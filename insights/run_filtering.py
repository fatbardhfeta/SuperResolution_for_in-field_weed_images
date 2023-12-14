import os
import json
import cv2


def apply_filter(image, filter_name):
    if filter_name == "median":
        return cv2.medianBlur(image, 5)  
    elif filter_name == "gaussian":
        return cv2.GaussianBlur(image, (9, 9), 0)  
    elif filter_name == "bilateral":
        return cv2.bilateralFilter(image, 15, 75, 75) 
    elif filter_name == "non-local_means":
        return cv2.fastNlMeansDenoisingColored(image, None, 15, 15, 7, 21) 
    else:
        print(f"Unknown filter name: {filter_name}")
        return image


def process_filter_images(filter_name, images_path, subdirs):
    print(f"Processing images for filter: {filter_name}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(images_path, subdir, "images")
        
        for image_filename in os.listdir(subdir_path):
            image_path = os.path.join(subdir_path, image_filename)
            original_image = cv2.imread(image_path)
            processed_image = apply_filter(original_image, filter_name)
            output_path = os.path.join(subdir_path, image_filename)
            cv2.imwrite(output_path, processed_image)
            #print(f"Processed and saved image: {output_path}")

def main():
    config_path = "configs/create_filtering_configuration.json"
    
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    
    subdirs = config["subdirs"]
    results = config["results"]
    
    for result in results:
        filter_name = result["filter_name"]
        images_path = result["images_path"]
        process_filter_images(filter_name, images_path, subdirs)

if __name__ == "__main__":
    main()
