import torch
import torch.nn as nn

@torch.no_grad()
def executeEvaluation(model, loader, device, scaler=None) -> float:
    """
    Oblicza średni loss walidacyjny dla modeli detekcyjnych (torchvision Faster R-CNN itp.).

    Uwaga: W trybie eval() modele detekcyjne zwracają listę predykcji, a nie straty.
    Dlatego na czas walidacji przełączamy model na train(), ale:
      - gradienty są WYŁĄCZONE (no_grad),
      - BatchNorm ustawiamy w eval() (żeby nie aktualizować running stats).
    Po wyliczeniu przywracamy pierwotny tryb modelu oraz stany BN.

    :param model: model detekcyjny (np. Faster R-CNN)
    :param loader: DataLoader walidacyjny (z collate_fn zwracającym listy obrazów i targetów)
    :param device: torch.device
    :param scaler: opcjonalnie GradScaler (dla zgodności z autocast)
    :return: średni val loss (float)
    """
    was_training = model.training
    # Tymczasowo train(), aby uzyskać dict strat
    model.train()

    # Ustaw BN w eval na czas walidacji
    bn_modules = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.training:
                bn_modules.append(m)
                m.eval()

    running_loss = 0.0
    n = 0

    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            loss_dict = model(images, targets)  # dict strat (dostępny w train mode)
            if isinstance(loss_dict, list):
                # Asekuracja – nie powinno się zdarzyć po przełączeniu na train()
                continue
            loss = sum(loss_dict.values())

        bs = len(images)
        running_loss += loss.item() * bs
        n += bs

    # Przywróć stany BN i tryb modelu
    for m in bn_modules:
        m.train()
    if not was_training:
        model.eval()

    return running_loss / max(n, 1)
