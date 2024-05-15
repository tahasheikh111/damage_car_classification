import os
import requests

# Function to remove background using Remove.bg API
def remove_background_api(input_path, output_path, api_key):
    # Ensure output folder exists
    os.makedirs(output_path, exist_ok=True)

    # Iterate over files in input folder
    for filename in os.listdir(input_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(input_path, filename)

            # Send request to Remove.bg API
            response = requests.post(
                'https://api.remove.bg/v1.0/removebg',
                files={'image_file': open(image_path, 'rb')},
                data={'size': 'auto'},
                headers={'X-Api-Key': api_key},
            )
            if response.status_code == requests.codes.ok:
                # Save output image with background removed
                output_image_path = os.path.join(output_path, f'no-bg_{filename}')
                with open(output_image_path, 'wb') as out:
                    out.write(response.content)
                print(f"Background removed successfully for {filename}")
            else:
                print(f"Error removing background for {filename}: {response.status_code}, {response.text}")

# Define input and output folders
input_folder = r'All_images\training\damage'
output_folder = r'All_images\training\damage_no_bg'

# Specify your Remove.bg API key
api_key = 'MkbNqVJZn9mSLGQhYJZ7T6by'

# Call function to remove background for images in input folder
remove_background_api(input_folder, output_folder, api_key)
