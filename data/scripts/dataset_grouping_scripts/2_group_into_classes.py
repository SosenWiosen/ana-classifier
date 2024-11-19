import os
import shutil

# specify source and destination paths
src_path = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D"
dst_path = "/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped"

classes = ["AC1", "AC2", "AC3", "AC4", "AC5", "AC6", "AC7", "AC8", "AC9", "AC10"]  # list all your classes here

def get_destination(dirpath, filename):
    path_parts = os.path.normpath(dirpath).split(os.path.sep)
    classname = None
    class_count = 0

    # Check every part of the path
    for part in path_parts:
        # Check every defined class
        for cls in classes:
            # If the class is in the path part, take it as the class name
            if cls in part:
                classname = cls
                class_count += 1
        # Break the loop if more than one class is found in a directory
        if class_count > 1:
            classname = None
            break
        if classname:
            break
            
    if classname:
        relative_dirpath = os.path.relpath(dirpath, src_path)
        new_name = f"{relative_dirpath.replace(os.path.sep, '_')}_{filename}"
        return f"{dst_path}/{classname}", new_name

    return None, None

def copy_files(src_path, dst_path):
    for dirpath, _, filenames in os.walk(src_path):
        #print('Checking in directory:', dirpath)

        for f in filenames:
            src = os.path.join(dirpath, f)
            # Check if file is a .tiff file
            if src.endswith(".tiff"):
                # Get the new destination folder and filename
                dst_folder, new_name = get_destination(dirpath, f)
                #print('Copying From:', src)
                #print('Copying To:', dst_folder)

                if dst_folder:
                    # Make sure the folder exists
                    os.makedirs(dst_folder, exist_ok=True)
                    dst = os.path.join(dst_folder, new_name)
                    shutil.copy2(src, dst)

# calling the copy function
copy_files(src_path, dst_path)