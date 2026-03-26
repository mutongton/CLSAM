import os
import SimpleITK as sitk
from tqdm import tqdm


def fix_nifti_sform(input_path, output_path):
    """
    读取 NIfTI 并重写（消除异常 sform）
    """
    img = sitk.ReadImage(input_path)

    # 关键操作：重新写一遍（SimpleITK会规范header）
    sitk.WriteImage(img, output_path)


def process_acdc_dataset(input_root, output_root):
    """
    按 ACDC 官方结构处理整个数据集
    """
    os.makedirs(output_root, exist_ok=True)

    patients = sorted(os.listdir(input_root))

    for patient in tqdm(patients, desc="Processing patients"):
        patient_in_dir = os.path.join(input_root, patient)
        patient_out_dir = os.path.join(output_root, patient)

        if not os.path.isdir(patient_in_dir):
            continue

        os.makedirs(patient_out_dir, exist_ok=True)

        files = os.listdir(patient_in_dir)

        for file in files:
            in_path = os.path.join(patient_in_dir, file)
            out_path = os.path.join(patient_out_dir, file)

            # 只处理 nii.gz
            if file.endswith(".nii.gz"):
                fix_nifti_sform(in_path, out_path)
            else:
                # 其他文件（Info.cfg）直接复制
                with open(in_path, "rb") as f_in:
                    with open(out_path, "wb") as f_out:
                        f_out.write(f_in.read())


if __name__ == "__main__":
    input_root = "ACDC/testing"
    output_root = "ACDC_fixed/testing"

    process_acdc_dataset(input_root, output_root)

    print("✅ Done! Fixed dataset saved to:", output_root)