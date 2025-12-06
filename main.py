from argparse import ArgumentParser

from utils import upload_envs
upload_envs()
from train.main import run as run_train
from test.main import run as run_test

def parseArgs():
    parser = ArgumentParser(description="Fine-tuning ResNet50")
    # Ścieżka do folderu z danymi podzielonymi klasami
    parser.add_argument("-m", "--mode", default="test", help="test|train", type=str, required=True)
    parser.add_argument("--clean-data", type=str, help="tak|nie", required=False, default="nie")
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArgs()

    if args.mode == 'train':
        run_train()
    elif args.mode == 'test':
        run_test()
    else:
        print('Usage: python main.py <train|test>')