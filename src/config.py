import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 512
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 40
#OPTIMIZER_PARAMS = {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9} # 옵티마이저 설정: SGD, 학습률, 모멘텀
#SCHEDULER_PARAMS = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2} # 학습률 스케줄러 설정

OPTIMIZER_PARAMS = {'type': 'Adam', 'lr': 0.001, 'weight_decay': 1e-4}  # Adam 최적화기 사용, 학습률 및 가중치 감쇠 설정으로 더 효과적인 최적화 가능
SCHEDULER_PARAMS = {'type': 'CosineAnnealingLR', 'T_max': 10}  # 학습률 스케줄러로 CosineAnnealing 사용, 주기적 학습률 조정

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
MODEL_NAME          = 'resnet18'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0] 
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
