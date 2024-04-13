import re

def group_folders():
    # Read the folder names from the text file
    with open("output.txt", "r", encoding="utf-8-sig") as file:
        folder_names = file.readlines()
        for folder_name in folder_names:
            folder_name = folder_name.strip()  # Remove leading/trailing whitespace and newline characters

    # Group the folder names by the portion before the second underscore
    grouped_folders = {}
    for folder_name in folder_names:
        match = re.search(r"k_(\d+)", str(folder_name))

        if match:
            first_part = match.group(1)
            if first_part not in grouped_folders:
                grouped_folders[first_part] = []
            grouped_folders[first_part].append(folder_name)

    # Print the grouped folders and their counts
    for first_part, folders in grouped_folders.items():
        folder_count = len(folders)
        print(f"{first_part} : {folder_count}")

if __name__ == "__main__":
    group_folders()