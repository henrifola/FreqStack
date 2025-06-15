import os
import shutil
from pathlib import Path

source_base_clean = Path("/home/ubuntu/Development/FreqNet_DeepfakeDetection/dataset/ForenSynths/progan")
source_base_perturbed = Path("/home/ubuntu/Development/FreqNet_DeepfakeDetection/dataset/perturbed-data/test/ForenSynths/progan")
output_test_perturbed = Path("/home/ubuntu/Development/FreqNet_DeepfakeDetection/dataset/EnsembleTestPerturbed/progan")

categories = ["car", "cat", "chair", "horse"]
subdirs = ["0_real", "1_fake"]

for category in categories:
    for subdir in subdirs:
        src_clean = source_base_clean / category / subdir
        src_perturbed = source_base_perturbed / category / subdir
        dst_test_perturbed = output_test_perturbed / category / subdir
        dst_test_perturbed.mkdir(parents=True, exist_ok=True)

        files_clean = sorted(os.listdir(src_clean))
        mid = len(files_clean) // 2
        test_clean_files = files_clean[:mid]

        perturbed_files = os.listdir(src_perturbed)

        for fname in test_clean_files:
            prefix = fname.split(".")[0].split("-")[0]  
            matched = next((f for f in perturbed_files if f.startswith(prefix)), None)

            if matched:
                shutil.copy2(src_perturbed / matched, dst_test_perturbed / matched)
            else:
                print(f"[!] No perturbed match found for: {prefix} in {category}/{subdir}")

print("Perturbed ensemble test dataset created with prefix matching.")
