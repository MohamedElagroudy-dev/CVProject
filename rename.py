import os

# folder path
folder = r"D:\caltech-101\brain"

# new base name
base_name = "brain"

# supported image formats
extensions = (".jpg", ".jpeg", ".png", ".bmp")

# get all images
files = []

for file in os.listdir(folder):
    if file.lower().endswith(extensions):
        files.append(file)

# sort files
files.sort()

# rename images
for i, file in enumerate(files, start=1):

    old_path = os.path.join(folder, file)

    ext = os.path.splitext(file)[1]

    new_name = f"{base_name}{i}{ext}"

    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)

    print(f"{file} -> {new_name}")

print("\nDone.")