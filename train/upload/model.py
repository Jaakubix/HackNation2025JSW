import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from transformers import AutoModel


# ===== Importy zgodne z różnymi wersjami torchvision =====
# Preferuj v2, fallback na klasyczny fpn
try:
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2 as maskrcnn_fn
    HAS_V2 = True
except Exception:
    from torchvision.models.detection import maskrcnn_resnet50_fpn as maskrcnn_fn
    HAS_V2 = False

# Bardzo lekki wariant (mobilenet 320) – opcjonalny last-resort (nie ma gotowego maskrcnn mnet 320 w old tv, ale to placeholder)
# Dla Mask R-CNN zazwyczaj używamy resnet50. Ignoruję mobilenet w tym przejściu dla uproszczenia,
# chyba że użytkownik wyraźnie prosił o mobile. Domyślnie Mask R-CNN R50.

# FastRCNNPredictor i MaskRCNNPredictor
try:
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor as _FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
except Exception:
    _FastRCNNPredictor = None
    MaskRCNNPredictor = None

# Wagi – warunkowo
try:
    if HAS_V2:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights as _Weights
        DEFAULT_WEIGHTS = _Weights.COCO_V1
    else:
        DEFAULT_WEIGHTS = None
except Exception:
    DEFAULT_WEIGHTS = None

TESTDATA_DIR = Path(os.getenv("TESTDATA_DIR", "./testdata"))
batch_size = int(os.environ['BATCH_SIZE'])
data_root = os.environ['DATA_ROOT']

class AnyModel:
    def __init__(self, model, feature_extractor=None):
        self.model = model
        self.feature_extractor = feature_extractor

    def load_state_dict(self, best_wts):
        self.model.load_state_dict(best_wts)

    def state_dict(self):
        return self.model.state_dict()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def modules(self):
        return self.model.modules()

    def train(self):
        self.model.train()

    def _compute_loss(self, imgs, label):
        with torch.no_grad():
            return self.model(imgs, label)

    def _compute_results(self, imgs):
        encoding = self.feature_extractor(imgs, return_tensors="pt")
        return self.model(**encoding)

    def _compute(self, imgs, label):
        # Akceptuj modele detekcyjne (Faster/Mask R-CNN)
        return self._compute_loss(imgs, label)

    def to(self, device):
        self.model.to(device)

    def __call__(self, imgs, label):
        return self._compute(imgs, label)


# ===== Prosty zamiennik FastRCNNPredictor (gdy brak w torchvision) =====
class SimpleFastRCNNPredictor(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, 1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas



# ===== Model =====
def _swap_predictor(model, num_classes: int):
    # 1. Głowica BBox
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    if _FastRCNNPredictor is not None:
        model.roi_heads.box_predictor = _FastRCNNPredictor(in_features, num_classes)
    else:
        model.roi_heads.box_predictor = SimpleFastRCNNPredictor(in_features, num_classes)

    # 2. Głowica Maski
    # Sprawdź czy model ma mask_predictor (Mask R-CNN powinien mieć)
    if hasattr(model.roi_heads, "mask_predictor") and MaskRCNNPredictor is not None:
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model


def getHFModel(model_name):
    model = AutoModel.from_pretrained(model_name)
    #feature_extractor = 
    return model

def getPretrainedModel(num_classes: int):
    """
    Buduje Mask R-CNN zmniejszony pod VRAM:
    - mniejszy rozmiar wejścia przez transform (min/max),
    - mniejsze batchy samplerów (RPN/RCNN),
    - mniej trenowanych warstw w backbone.
    """

    # Parametry „odchudzające”
    kwargs_common = dict(
        min_size=480,               # krótszy bok po przeskalowaniu przez transform modelu
        max_size=640,               # dłuższy bok (domyślnie nawet 1333)
        rpn_pre_nms_top_n_train=200,
        rpn_post_nms_top_n_train=100,
        rpn_pre_nms_top_n_test=100,
        rpn_post_nms_top_n_test=50,
        rpn_batch_size_per_image=64,   # domyślnie 256
        box_batch_size_per_image=128,  # domyślnie 512
        trainable_backbone_layers=2,   # mniej VRAM
    )

    model_name = os.environ['MODEL']

    if ("/" in model_name):
        model = getHFModel(model_name)
    else:
        # Zawsze Mask R-CNN resnet50
        if DEFAULT_WEIGHTS is not None:
            model = maskrcnn_fn(weights=DEFAULT_WEIGHTS, **kwargs_common)
        else:
            # starsze API
            model = maskrcnn_fn(pretrained=True, **kwargs_common)
            
        model = _swap_predictor(model, num_classes)

    if bool(int(os.environ["FREEZE_BACKBONE"])):
        model.backbone.requires_grad_(False)

    return model


# ===== Zapis modelu/metryk =====
def saveModel(model, best_wts, device, best_val_loss: float, train_losses: list, val_losses: list) -> bool:
    model_name = os.environ["TARGET_MODEL_NAME"] + "_" + os.environ['MODEL']
    output_dir = Path(os.environ['OUTPUT_DIR'])  # finetuned_models
    model_path = output_dir / f"{model_name}_model.pth"
    metrics_path = output_dir / f"{model_name}_metrics.json"  # Metryki w tym samym folderze co model
    plot_path = output_dir / f"{model_name}_loss_vs_epochs.png"  # Wykres też w tym samym folderze

    output_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_wts)
    model.eval()

    if model_path.exists() and metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                old_metrics = json.load(f)
                old_best = old_metrics.get("best_val_loss", float("inf"))
        except Exception as e:
            print(f"Nie można odczytać poprzednich metryk: {e}")
            old_best = float("inf")

        if best_val_loss >= old_best - 1e-12:
            print(f"Nowy model (val_loss={best_val_loss:.4f}) nie jest lepszy od starego (val_loss={old_best:.4f}). Nie zapisano.")
            return False
        else:
            print(f"Nowy model (val_loss={best_val_loss:.4f}) jest lepszy niż stary (val_loss={old_best:.4f}). Zapisuję nowy model.")
    else:
        print("Nie znaleziono poprzedniego modelu lub metryk. Zapisuję nowy model.")

    torch.save(model.state_dict(), model_path)

    # Zapisz metryki w folderze z modelem
    metrics_data = {
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs": len(train_losses),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Zapisano model do {model_path}")
    print(f"Zapisano metryki do {metrics_path}")

    # Wykres w folderze z modelem
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

    print(f"Zapisano wykres do {plot_path}")
    return True
