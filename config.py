import os 
from os.path import join, dirname
from dotenv import load_dotenv
load_dotenv(join(dirname(__file__), '.env'))

LOG_DIR = join(dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

CIFAR_ROOT = os.getenv('CIFAR_ROOT')
if CIFAR_ROOT is None:
    raise Exception('.envファイルにCIFAR_ROOTの設定が見つかりません')

MODEL_OUTPUT = os.getenv('MODEL_OUTPUT')
os.makedirs(MODEL_OUTPUT, exist_ok=True)
if MODEL_OUTPUT is None:
    raise Exception('.envファイルにMODEL_OUTPUTの設定が見つかりません')