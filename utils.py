import os
import shutil
import glob


def rename_files():
    dataset_dir = r"dataset"
    filenames = glob.glob(os.path.join(dataset_dir, "**", "*.tif"), recursive=True)
    for filename in filenames:
        new_file_name = filename + "f"
        shutil.move(filename, new_file_name)
        # print(filename)


if __name__ == "__main__":
    rename_files()
