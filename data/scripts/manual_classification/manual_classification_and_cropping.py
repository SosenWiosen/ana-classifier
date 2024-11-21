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

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            return json.load(file)
    return {}

def save_progress(file_map, progress_file):
    with open(progress_file, 'w') as file:
        json.dump(file_map, file, indent=4)

def classify_images(original_dir, new_dir, progress_file):
    file_map = load_progress(progress_file)
    available_classes = ["AC1", "AC2", "AC3", "AC4"]
    index_counter = {}
    allowed_extensions = {".tiff", ".png"}


    for subdir, _, files in os.walk(original_dir):
        for file in files:
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
                # print("Cropping image...")
                # print( "filepath: ", filepath)
                # Use the crop function to get a cropped image as a PIL Image object
                cropped_image = crop(filepath)
                # print("Cropped image: ", cropped_image.size)
                if not cropped_image:
                    break
                
                cropped_new_filename = f"{image_class}_{method}_{index_counter[index_key]}_crop{crop_index}.png"
                cropped_new_file_path = os.path.join(cropped_dir, cropped_new_filename)
                
                # Save the cropped image
                cropped_image.save(cropped_new_file_path)
                
                # Update the file map
                file_map[os.path.relpath(filepath, original_dir)] = {
                    "non-cropped": os.path.relpath(filepath, new_dir),
                    "cropped": os.path.relpath(cropped_new_file_path, new_dir),
                    "weird": not image_doubts
                }
                
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