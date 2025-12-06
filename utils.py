import os

from dotenv import load_dotenv
import torch

def upload_envs():
    load_dotenv()

device_prompted = False
def selectDevice():
    global device_prompted
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not device_prompted:
        device_prompted = True
        print(f"Device selected: {device.type}")

    return device

def setSeed(seed: int = 0):
    seed = seed | int(os.environ["SEED"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # if selectDevice().type == "cuda" and bool(os.environ["USE_DETERMINISTIC"]):
    #     torch.use_deterministic_algorithms(True)
    # else:
    #     torch.use_deterministic_algorithms(False)
