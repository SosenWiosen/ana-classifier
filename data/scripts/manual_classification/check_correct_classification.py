import json

def load_data(filepath):
    """ Load JSON data from a file. """
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def check_class_in_filename(data):
    """ Check if the class is part of the original filename for each entry in data. """
    count = 0
    wrongCount = 0
    for original_file, file_info in data.items():
        non_cropped_file = file_info['non-cropped']
        file_class = non_cropped_file.split('/')[-2]
        count+=1

        if file_class not in original_file:
            print(f"Class '{file_class}' is NOT part of the original filename '{original_file}'.")
            wrongCount+=1
    print(f"Checked {count} entries.")
    print(f"Found {wrongCount} entries with incorrect class in the filename.")

# Load the data from a JSON file
file_path = '/Users/sosen/UniProjects/eng-thesis/data/manual/file_map.json'
data = load_data(file_path)

# Execute the class checking function
check_class_in_filename(data)