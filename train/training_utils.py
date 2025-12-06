import os

def set_optimizer(model, learning_rate, weight_decay):
    # Adafactor, Adadelta, Adagrad, Adam, Adamax, AdamW, ASGD, LBFGS,
    # NAdam, Optimizer, RAdam, RMSprop, Rprop, SGD, SparseAdam
    print(f"Chosen optimizer is {os.environ['OPTIMIZER']}")
    match os.environ["OPTIMIZER"]:
        case "SGD":
            from torch.optim import SGD
            return SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        case "Adam":
            from torch.optim import Adam
            return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        case "AdaGrad":
            from torch.optim import Adagrad
            return Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        case "RMSProp":
            from torch.optim import RMSprop
            return RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        case "AdamW":
            from torch.optim import AdamW
            return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        case "Adamax":
            from torch.optim import Adamax
            return Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        case "Rprop":
            from torch.optim import Rprop
            return Rprop(model.parameters, lr=learning_rate)
        case "Adadelta":
            from torch.optim import Adadelta
            return Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        case "Adafactor":
            from torch.optim import Adafactor
            return Adafactor(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        case "ASGD":
            from torch.optim import ASGD
            return ASGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        case "LBFGS":
            from torch.optim import LBFGS
            return LBFGS(model.parameters(), lr=learning_rate)
        case "NAdam":
            from torch.optim import NAdam
            return NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        case "RAdam":
            from torch.optim import RAdam
            return RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        case "SparseAdam":
            from torch.optim import SparseAdam
            return SparseAdam(model.parameters(), lr=learning_rate)
        case _:
            from torch.optim import SGD
            return SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def set_scheduler(optimizer, T_max, eta_min):
    # AVAILABLE SCHEDULERS: LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ConstantLR, LinearLR,
    # ExponentialLR, SequentialLR, CosineAnnealingLR, ChainedScheduler, ReduceLROnPlateau, CyclicLR,
    # CosineAnnealingWarmRestarts, OneCycleLR, PolynomialLR, LRScheduler
    print(f"Chosen scheduler is {os.environ['SCHEDULER']}")
    match os.environ["SCHEDULER"]:
        case "CosineAnnealingLR":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        case "CosineAnnealingWarmRestarts":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            return CosineAnnealingWarmRestarts(optimizer, T_0=T_max, T_mult=1, eta_min=eta_min)
        case "LambdaLR":
            from torch.optim.lr_scheduler import LambdaLR
            return LambdaLR(optimizer, lr_lambda=lambda step: 1 - step / T_max)
        case "StepLR":
            from torch.optim.lr_scheduler import StepLR
            return StepLR(optimizer, step_size=T_max, gamma=0.1)
        case _:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
