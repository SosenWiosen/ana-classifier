import os
import shutil
import json
import tkinter as tk
from PIL import Image
from image_cropper import ImageCropper
from matplotlib import pyplot as plt
from termcolor import colored


def crop(filepath):
    def return_cropped_image(cropped_img):
        nonlocal cropped_image
        cropped_image = cropped_img
        image_cropper.root.quit()

    cropped_image = None
    root = tk.Tk()
    image_cropper = ImageCropper(root, return_cropped_image)
    image_cropper.center_window_on_primary_display(1200,1000)
    image_cropper.open_image(filepath)
    root.mainloop()

    # root.destroy()

    return cropped_image

def save_progress(file_map, progress_file):
    with open(progress_file, 'w') as file:
        json.dump(file_map, file, indent=4)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            return json.load(file)
    return {}
def load_index_counter(file_map):
    index_counter = {}
    for file_info in file_map.values():
        if isinstance(file_info, dict):
            # Extract "non-cropped" path and determine class and method
            non_cropped_path = file_info.get("non-cropped", "")
            if non_cropped_path:
                # Split the path to extract method dir and class
                path_parts = non_cropped_path.split(os.sep)
                if len(path_parts) > 3:  # Ensure the path has enough depth
                    method_dir = path_parts[-3]  # 'STED' or 'NON-STED'
                    image_class = path_parts[-2]  # Class name
                    key = f"{image_class}_{method_dir}"
                    # Extract the last used index from the filename
                    filename = path_parts[-1]  # e.g., "AC1_STED_10.png"
                    try:
                        index = int(filename.split('_')[-1].split('.')[0])  # Extract the numeric portion
                        index_counter[key] = max(index_counter.get(key, 0), index)
                    except (IndexError, ValueError):
                        continue  # Ignore files with improperly formatted filenames
    return index_counter
def classify_images(original_dir, new_dir, progress_file):
    # Define available classes
    available_classes = ["AC1", "AC2", "AC3", "AC4", "AC5", "AC6", "AC7", "AC8", "AC9", "AC10", "TISSUE"]

    # Load progress
    file_map = load_progress(progress_file)

    # Dynamically add existing classes from the progress file
    for file_data in file_map.values():
        if isinstance(file_data, dict):  # Ensure it has the structured dictionary
            cropped_paths = file_data.get("cropped", [])
            if isinstance(cropped_paths, list):  # Verify it's a list
                # Loop through all cropped paths to extract class directories
                for path in cropped_paths:
                    path_parts = path.split(os.sep)  # Safely split each path
                    if len(path_parts) > 2:  # Ensure path is deep enough to extract the class
                        class_from_progress = path_parts[-3]  # Extract the class
                        if (
                            class_from_progress not in ["STED", "NON-STED"] and
                            class_from_progress not in available_classes
                        ):
                            available_classes.append(class_from_progress)

    index_counter = load_index_counter(file_map)
    allowed_extensions = {".tiff", ".png"}

    for subdir, _, files in os.walk(original_dir):
        for file in sorted(files):
            filepath = os.path.join(subdir, file)
            relative_path = os.path.relpath(filepath, original_dir)
            # Skip files that have already been processed
            if relative_path in file_map:
                print(f"Skipping already processed file: {relative_path}")
                continue
            # Check for allowable image extensions
            if not os.path.splitext(file)[1].lower() in allowed_extensions:
                print(f"Skipping non-image file: {relative_path}")
                continue

            print(colored(f"Processing file: {file} \nRelative path: {relative_path}", 'green', attrs=['bold']))
            Image.open(filepath).show()

            # Ask for imaging method
            while True:
                method_input = input("\nIs this image STED (1/y) or normal (2/n)? ").strip().lower()
                if method_input in ('1', 'y', 'sted'):
                    method = "STED"
                    method_dir = "STED"
                    break
                elif method_input in ('2', 'n', 'normal'):
                    method = "normal"
                    method_dir = "NON-STED"
                    break
                else:
                    print("Invalid choice. Please enter '1', '2', 'y', or 'n'.")

            # Ask for image class
            while True:
                class_prompt = "\n".join([f"{i+1}. {cls}" for i, cls in enumerate(available_classes)])
                other_new_start_index = len(available_classes) + 1
                print("\nChoose a class: \n" + class_prompt + f"\n{other_new_start_index}. Other")
                print(f"{other_new_start_index + 1}. New Class")
                class_choice = input("Class (number): ").strip()
                if class_choice.isdigit():
                    class_index = int(class_choice) - 1
                    if class_index < len(available_classes):
                        image_class = available_classes[class_index]
                        break
                    elif class_index == len(available_classes):
                        image_class = "Other"
                        break
                    elif class_index == len(available_classes) + 1:
                        new_class = input("Enter new class name: ").strip()
                        if new_class not in available_classes:
                            available_classes.append(new_class)
                        image_class = new_class
                        break
                print("Invalid choice, please try again.")

            while True:
                image_doubts_input = input("\nIs this image ok? or no (y/n)").strip().lower()
                if image_doubts_input in ('1', 'y'):
                    image_doubts = True
                    break
                else:
                    image_doubts = False
                    break

            # Prepare the destination directory
            dest_dir = os.path.join(new_dir, "NON-CROPPED", method_dir, image_class)
            os.makedirs(dest_dir, exist_ok=True)

            # Prepare the new filename
            index_key = f"{image_class}_{method_dir}"
            if index_key not in index_counter:
                index_counter[index_key] = 0
            index_counter[index_key] += 1
            new_filename = f"{image_class}_{method}_{index_counter[index_key]}.png"

            # Copy the file to the new directory
            new_file_path = os.path.join(dest_dir, new_filename)
            shutil.copy(filepath, new_file_path)




            # Initialize or update the file_map entry
            if relative_path not in file_map or not isinstance(file_map[relative_path], dict):
                # If not initialized or if the data structure is inconsistent, initialize it
                file_map[relative_path] = {
                    "non-cropped": os.path.relpath(new_file_path, new_dir),
                    "cropped": [],
                    "weird": not image_doubts
                }

            # Cropping
            should_crop = True
            cropped_dir = os.path.join(new_dir, "CROPPED", method_dir, image_class)
            os.makedirs(cropped_dir, exist_ok=True)
            crop_index = 1
            while should_crop:
                # Use the crop function to get a cropped image as a PIL Image object
                cropped_image = crop(filepath)
                if not cropped_image:
                    break

                cropped_new_filename = f"{image_class}_{method}_{index_counter[index_key]}_crop{crop_index}.png"
                cropped_new_file_path = os.path.join(cropped_dir, cropped_new_filename)

                # Save the cropped image
                cropped_image.save(cropped_new_file_path)


                # Append the new cropped file path to the "cropped" list
                file_map[relative_path]["cropped"].append(os.path.relpath(cropped_new_file_path, new_dir))
                crop_index += 1
                should_crop_input = input("Do you want to crop more objects from this image (y/n)? ").strip().lower()
                should_crop = should_crop_input in ('1', 'y')

            # Save progress after processing each file
            save_progress(file_map, progress_file)

    # Final save to ensure everything is recorded
    save_progress(file_map, progress_file)
    print(f"Classification and cropping complete. File map saved at: {progress_file}")

if __name__ == "__main__":
    original_directory = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D"
    new_directory = "/Users/sosen/UniProjects/eng-thesis/data/manual"
    progress_file = os.path.join(new_directory, "file_map.json")
    
    classify_images(original_directory, new_directory, progress_file)