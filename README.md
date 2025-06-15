# FreqStack: Robust Deepfake Detection with Global and Local Frequency Cues

**FreqStack** is a deepfake detection system designed to remain effective under real-world image degradations such as blur, noise, and JPEG compression. It combines two complementary branches:
- A **global frequency branch** that analyzes Fourier-transformed images (based on FreqNet).
- A **local wavelet-based branch** that captures localized artifacts and high-frequency inconsistencies.
These are fused using a lightweight ensemble meta-learner.

## Project Structure

- `FreqNet_DeepfakeDetection/` – Submodule based on original FreqNet, adapted for global frequency analysis.
- `LoalFreq/` – Custom wavelet-based CNN for local feature extraction.
- `Ensemble_Learner/` – Ensemble meta-learner that combines predictions from both branches.
- `test.py` – Unified test script for evaluating each component and the ensemble.

## Datasets

This project uses a custom subset of the **ForenSynths** dataset, with:
- 8 GAN classes: `progan`, `stylegan`, `stylegan2`, `biggan`, `cyclegan`, `stargan`, `gaugan`, `deepfake`
- 4 trained object classes: `car`, `cat`, `chair`, `horse`
- Perturbation variants: blur, noise, JPEG compression, and random combinations (50% chance per image)

## Testing

### Test Local Branch

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --model_path /path/to/llnet_checkpoint.pth \
  --dataroot /path/to/dataset \
  --classes car cat chair horse
```

### Test Global Branch

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --model_path /path/to/freqnet_checkpoint.pth \
  --dataroot /path/to/ForenSynths \
  --classes car,cat,chair,horse \
  --batch_size 32
```

### Test Ensemble (Full FreqStack)

```bash
python Ensemble_Learner/test.py
```

The `test.py` script loads both branches and passes predictions to a MetaLearner that fuses outputs and evaluates performance on different perturbation conditions.




> Folaasen, Henriette. "Enhancing Deepfake Detection Under Distortions with Global and Local Frequency Cues." 