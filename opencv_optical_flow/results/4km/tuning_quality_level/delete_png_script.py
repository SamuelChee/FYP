import os
import glob

def delete_png_files(directory):
    # Search for PNG files recursively
    pattern = os.path.join(directory, '**/*.png')
    png_files = glob.glob(pattern, recursive=True)

    # Delete each PNG file
    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except OSError as e:
            print(f"Error deleting file: {file_path} - {e}")

# Specify the directory path where you want to delete PNG files
directory_path = '.'
if __name__ == "__main__":
    # Call the function to delete PNG files within the directory
    delete_png_files(directory_path)