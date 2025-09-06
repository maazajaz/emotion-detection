import os

# The path to your dataset folder
DATASET_PATH = "custom_dataset"

print(f"--- Checking contents of '{DATASET_PATH}' ---")

# Check if the main dataset folder exists
if not os.path.exists(DATASET_PATH):
    print(f"ERROR: The main folder '{DATASET_PATH}' does not exist!")
else:
    # Get all the sub-folders (e.g., '0_bored', '1_not_looking')
    sub_folders = os.listdir(DATASET_PATH)

    if not sub_folders:
        print("The 'custom_dataset' folder is empty.")
    else:
        print(f"Found sub-folders: {sub_folders}\n")

        # Loop through each sub-folder
        for folder in sub_folders:
            folder_path = os.path.join(DATASET_PATH, folder)

            # Check if it's actually a folder
            if os.path.isdir(folder_path):
                # Get all the files inside the sub-folder
                files = os.listdir(folder_path)
                print(f"--- In folder '{folder}':")
                if not files:
                    print("    -> This folder is EMPTY.")
                else:
                    print(f"    -> Found {len(files)} files. The first 5 are:")
                    # Print the first 5 files found
                    for file_name in files[:5]:
                        print(f"       - {file_name}")
            print("-" * 20)

print("\n--- Check complete ---")