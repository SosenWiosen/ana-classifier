import zipfile
import shutil
from pathlib import Path

def process_zip_file(zip_path, destination_dir):
    # Extract ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        extract_path = zip_path.with_suffix('')
        zip_ref.extractall(extract_path)
        print(f'Extracted {zip_path} to {extract_path}')
        
        # Copy TIFF files from extracted folder
        copy_tiff_files(extract_path, destination_dir)
        
        # Try to remove the folder after copying TIFF files; ignore errors
        try:
            shutil.rmtree(extract_path)
            print(f'Deleted {extract_path}')
        except Exception as e:
            print(f"Could not delete {extract_path}: {e}")

def copy_tiff_files(source_dir, destination_dir):
    # Recurse through the source directory for TIFF files
    for item in Path(source_dir).rglob('*'):
        if item.is_file() and item.suffix in ['.tiff', '.tif']:
            relative_path = item.relative_to(source_dir)
            destination_path = Path(destination_dir) / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(item, destination_path)
            print(f'Copied {item} to {destination_path}')

def main(zips_dir, dest_dir):
    zips_dir = Path(zips_dir).expanduser()
    dest_dir = Path(dest_dir).expanduser()

    dest_dir.mkdir(parents=True, exist_ok=True)

    for zip_file in zips_dir.glob('*.zip'):
        process_zip_file(zip_file, dest_dir)
    print("Processing complete.")

if __name__ == "__main__":
    zips_dir = '/Users/sosen/UniProjects/eng-thesis/data/data-compressed'
    destination_dir = '/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D'
    main(zips_dir, destination_dir)