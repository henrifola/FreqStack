run
CUDA_VISIBLE_DEVICES=0 python train.py \
  --isTrain \
  --name 4class-llnet-car-cat-chair-horse-sgd \
  --dataroot /home/ubuntu/Development/FreqNet_DeepfakeDetection/dataset \
  --classes car cat chair horse \
  --batch_size 32 \
  --delr_freq 10 \
  --lr 0.01 \
  --niter 85 \
  --optim sgd


CUDA_VISIBLE_DEVICES=0 python train.py \
  --isTrain \
  --name 4class-llnet-car-cat-chair-horse-sgd \
  --dataroot /home/ubuntu/Development/FreqNet_DeepfakeDetection/dataset \
  --classes car cat chair horse \
  --batch_size 32 \
  --delr_freq 10 \
  --lr 0.01 \
  --niter 85 \
  --optim sgd \
  --weight_decay 1e-4



  CUDA_VISIBLE_DEVICES=0 python train.py \
  --name 4class-resnet-car-cat-chair-horse \
  --dataroot /home/ubuntu/Development/FreqNet_DeepfakeDetection/dataset --classes car,cat,chair,horse \
  --batch_size 32 \
  --delr_freq 10 \
  --lr 0.001 \
  --niter 85 \

or

CUDA_VISIBLE_DEVICES=0 python train.py \
  --isTrain \
  --name 4class-llnet-car-cat-chair-horse \
  --dataroot /home/ubuntu/Development/FreqNet_DeepfakeDetection/dataset \
  --classes car cat chair horse \
  --batch_size 32 \
  --delr_freq 10 \
  --lr 0.0003 \
  --niter 85 \
  --beta1 0.9 \
  --momentum 0.9 \
  --weight_decay 1e-4 \
  --optim adam


when train