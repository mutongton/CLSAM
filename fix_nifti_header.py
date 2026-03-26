import os
import SimpleITK as sitk

dataset_path = "ACDC"

folders = ["imagesTs", "labelsTs"]

for folder in folders:
    folder_path = os.path.join(dataset_path, folder)

    for file in os.listdir(folder_path):
        if file.endswith(".nii.gz"):
            file_path = os.path.join(folder_path, file)

            print("Processing:", file_path)

            # 读取
            img = sitk.ReadImage(file_path)

            # 重新写入
            sitk.WriteImage(img, file_path)

print("All files rewritten successfully.")