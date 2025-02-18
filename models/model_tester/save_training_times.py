import json


def save_training_times(times, file_path):
    # Create a dictionary to store times and their sum
    timing_data = {
        'epoch_times': times,
        'total_time': sum(times)
    }

    # Serialize the dictionary to a JSON formatted string with indentation for readability
    timing_json = json.dumps(timing_data, indent=4)

    # Write the JSON string to a file
    with open(file_path, 'w') as file:
        file.write(timing_json)
