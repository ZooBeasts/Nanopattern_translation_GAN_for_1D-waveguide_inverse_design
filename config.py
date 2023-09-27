import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "circle_images"
VAL_DIR = "circle_imagestest"
LEARNING_RATE = 2e-4
BATCH_SIZE = 64
NUM_WORKERS = 0
IMAGE_SIZE = 64
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

