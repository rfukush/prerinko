"""
必要なものをインポート / Import libraries necessary
"""
import os  # システム、パス設定などのため
import sys
from os.path import join
import time
import datetime  # 日時関連のもののため
import logging  # 実行記録を残すため
import argparse  # スクリプトのオプションを追加するため
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import torch  # PyTorchフレームワーク
import torch.nn as nn  # PyTorchのニューラルネットワークモジュール
import torch.optim as optim  # PyTorchの最適化モジュール
import torchvision  # PyTorchのComputer Vision用フレームワーク
import torchvision.transforms as transforms  # 画像処理のモジュール
from torchvision.models import vgg19

from config import LOG_DIR, CIFAR_ROOT, MODEL_OUTPUT

""" 
Step 1. ログ設定 / Setup logging

    Example use:

        logger.info('Start!')
"""
now = datetime.datetime.now()
date = now.strftime("%Y%m%d")
timestamp = now.timestamp()
log_path = join(LOG_DIR, date)
os.makedirs(log_path, exist_ok=True)
filename = join(log_path, f'{int(timestamp)}_{os.getpid()}.log')
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

fh = logging.FileHandler(filename=filename)
fh.setFormatter(fmt)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(sh)

"""
Step 2. オプション設定 / Setup args

    Run: python main.py -i 4
    
    Code: i = args.i  # i == 4
"""

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=int, default=0, help='Specify which GPU to use. Defaults to 0.')
parser.add_argument('--num_epochs', type=int, default=3, help='Specify number of training epochs. Defaults to 3.')
parser.add_argument('--batch_size', type=int, default=4, help='Specify training batch size. Defaults to 4.')
parser.add_argument('--lr', type=float, default=1e-3, help='Specify training learning rate. Defaults to 0.001.')
parser.add_argument('--use_cpu', action='store_true', help='Use CPU for training. Defaults to false.')
parser.add_argument('--do_full_training', action='store_true', help='Do full training. Defaults to false.')
parser.add_argument('--do_eval_only', action='store_true', help='Only do evaluation. Defaults to false.')
args = parser.parse_args()

"""
Step 3. 訓練パラメータ設定 / Setup training parameters
"""

# GPU Settings
if args.use_cpu:
    DEVICE = torch.device('cpu')
    logger.info('CPU使います...')
else:
    if torch.cuda.is_available():
        DEVICE = torch.device(f'cuda:{args.gpu}')
        logger.info(f'{args.gpu}番GPU使います！')
    else:
        DEVICE = torch.device('cpu')
        logger.info('GPU使えないからCPU使います...')

# Hyperparamter Settings
NUM_EPOCHS = args.num_epochs if args.num_epochs > 0 else 3
if args.num_epochs <= 0:
    logger.warning('Num epochsは1以上の数字を指定してください')

BATCH_SIZE = args.batch_size if args.batch_size > 0 else 4
if args.batch_size <= 0:
    logger.warning('Batch sizeは1以上の数字を指定してください')

LR = args.lr if args.lr > 0 else 1e-3
if args.lr <= 0:
    logger.warning('Learning rateは0より大きい値を指定してください')

"""
Step 4. データセット準備 / Prepare dataset
"""

logger.info('データの準備開始...')
start = time.time()

TRANSFORM = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

trainset = torchvision.datasets.CIFAR10(
                root=CIFAR_ROOT, 
                train=True, 
                download=True, 
                transform=TRANSFORM,
                )

testset = torchvision.datasets.CIFAR10(
                root=CIFAR_ROOT, 
                train=False,
                download=True, 
                transform=TRANSFORM,
                )

train_loader = torch.utils.data.DataLoader(
                trainset, 
                batch_size=BATCH_SIZE,
                shuffle=True, 
                num_workers=2,
                )

test_loader = torch.utils.data.DataLoader(
                testset, 
                batch_size=BATCH_SIZE,
                shuffle=False, 
                num_workers=2,
                )

logger.info(f'データの準備完了！ | Time spent: {time.time() - start:.1f}s')

"""
Step 5. モデル・最適化アルゴリズム・損失関数準備 / Prepare model, optimizer, and loss function
"""

class MyVGG19(nn.Module):  # モデルを自由に作れる！
    def __init__(self) -> None:
        super().__init__()
        self.base_model = vgg19()

    def forward(self, inputs):
        return self.base_model(inputs)

model = MyVGG19().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()  # 分類タスクでよく使われる損失関数

"""
Step 6. 訓練ループ / Training loop
"""
if not args.do_eval_only:
    logger.info('訓練開始！')
    start = time.time()
    with logging_redirect_tqdm(loggers=[logger]):
        for epoch in range(NUM_EPOCHS):

            running_loss = 0.0
            for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                if i >= len(train_loader)//4 and not args.do_full_training:  # Demoでは途中でやめます
                    logger.info(f"Demoなので、step {i} でやめます。")
                    # logger.info("please note that trainloader is ITERATOR and so usually can NOT be canceled at middle.")
                    break
                
                inputs, labels = map(lambda x: x.to(DEVICE), data)  # data: [(inputs, labels), (inputs, labels), ...]
                                                                    # Move data to device (The model and data must be on the same device)
                
                optimizer.zero_grad()  # Clear the accumulated gradients

                outputs = model(inputs)  # Get the outputs in logits
                loss = criterion(outputs, labels)  # Get the training loss from the loss function
                loss.backward()  # Back propagation
                optimizer.step()  # Update gradients

                running_loss += loss.item()  # Accumulate total loss for every 100 steps
                if (i + 1) % 200 == 0:  # Log every 100 steps and clear running_loss
                    logger.info(f'epoch {epoch + 1} | step {i + 1} / {len(train_loader)} | loss {running_loss / 100:.3f}')
                    running_loss = 0.0

            model_path = join(MODEL_OUTPUT, f'cifar10_vgg19_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)  # Save checkpoints every epoch

    logger.info(f'Training is complete. | Time spent: {time.time() - start:.1f}s')

"""
Step 7. 性能評価 / Model evaluation
"""
if args.do_eval_only:
    model.load_state_dict(torch.load(join(MODEL_OUTPUT, f'cifar10_vgg19_epoch{NUM_EPOCHS}.pth')))

logger.info(f'Start model evaluation.')
start = time.time()
model.eval()  # No gradients for since we're not training, we don't need to calculate the gradients for our outputs
num_correct = 0
total = 0

with logging_redirect_tqdm(loggers=[logger]):
    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, labels = map(lambda x: x.to(DEVICE), data)

            outputs = model(inputs)  # Shape: batch_size x num_cls
            prediction = outputs.softmax(1)
            predicted = prediction.argmax(1)  # Get class id of prediction with highest score
            # score = prediction[predicted].item()

            total += labels.size(0)
            num_correct += (predicted == labels).sum().item()

logger.info(f'Accuracy on {total} test images: {100 * num_correct / total:.2f}%. | Time spent: {time.time() - start:.1f}s')