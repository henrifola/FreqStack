# LoalFreq: Local Frequency-Based Deepfake Detection Branch

This module implements the **local branch** of the FreqStack deepfake detection system. It focuses on detecting localized, high-frequency artifacts using wavelet-decomposed color channels and a patch-based CNN architecture.

---

## 🔧 Training Commands

You can train the model using different optimizers and learning rate configurations.

###  SGD 

Basic version:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --isTrain \
  --name 4class-llnet-car-cat-chair-horse-sgd \
  --dataroot /path/to/dataset \
  --classes car cat chair horse \
  --batch_size 32 \
  --delr_freq 10 \
  --lr 0.01 \
  --niter 85 \
  --optim sgd
```

With weight decay:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --isTrain \
  --name 4class-llnet-car-cat-chair-horse-sgd \
  --dataroot /path/to/dataset \
  --classes car cat chair horse \
  --batch_size 32 \
  --delr_freq 10 \
  --lr 0.01 \
  --niter 85 \
  --optim sgd \
  --weight_decay 1e-4
```

---

### ⚙️ Adam Optimizer 

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --isTrain \
  --name 4class-llnet-car-cat-chair-horse \
  --dataroot /path/to/dataset \
  --classes car cat chair horse \
  --batch_size 32 \
  --delr_freq 10 \
  --lr 0.0003 \
  --niter 85 \
  --beta1 0.9 \
  --momentum 0.9 \
  --weight_decay 1e-4 \
  --optim adam
```

---

### 🧪 ResNet Baseline (Comparison)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --name 4class-resnet-car-cat-chair-horse \
  --dataroot /path/to/dataset \
  --classes car,cat,chair,horse \
  --batch_size 32 \
  --delr_freq 10 \
  --lr 0.001 \
  --niter 85
```

---

## 🗂 Dataset

Uses a subset of **ForenSynths**, focused on:
- Classes: `car`, `cat`, `chair`, `horse`
- Dataroot: `/path/to/dataset`

Make sure your dataset folder contains appropriate `0_real` and `1_fake` directories for each class.

---

## 💬 Notes

- `--delr_freq` refers to how often logs are printed and possibly checkpoints are saved.
- All training logs and model checkpoints will be stored under `checkpoints/{experiment_name}/`.

---

## 📤 Outputs

Final models are saved as:

```
checkpoints/{experiment_name}/epoch_{EPOCH}_model.pth
```

These models are later used in the ensemble evaluation inside the main `FreqStack` wrapper.