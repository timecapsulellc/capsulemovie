import os
import glob

def replace_in_files(directory, old_str, new_str):
    for filepath in glob.glob(f"{directory}/**/*.py", recursive=True):
        with open(filepath, 'r') as file:
            content = file.read()
        content = content.replace(old_str, new_str)
        with open(filepath, 'w') as file:
            file.write(content)

if __name__ == "__main__":
    replace_in_files(".", "capsule_movie_core", "capsule_movie_core")