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

    index_counter = {}
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

            # Save file paths to the map
            file_map[relative_path] = os.path.relpath(new_file_path, new_dir)

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

                file_relative_path = os.path.relpath(filepath, original_dir)

                # Initialize `file_map[file_relative_path]` as a dictionary if it doesn't exist
                if file_relative_path not in file_map or not isinstance(file_map[file_relative_path], dict):
                    file_map[file_relative_path] = {
                        "non-cropped": os.path.relpath(filepath, new_dir),
                        "cropped": [],  # Start with an empty list for cropped images
                        "weird": not image_doubts
                    }

                # Append the new cropped file path to the "cropped" list
                file_map[file_relative_path]["cropped"].append(os.path.relpath(cropped_new_file_path, new_dir))
                crop_index += 1
                should_crop = input("Do you want to crop more objects from this image (y/n)? ").strip().lower() == 'y'

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