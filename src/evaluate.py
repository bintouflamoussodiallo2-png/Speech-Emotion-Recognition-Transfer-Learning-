import torch 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct += (outputs.argamx(1) == labels).sum().item()
            total += labels.size(0)

    return correct / total

def get_predictions(model, loader, device):
    model.eval()
    list_pred, list_labels = [], []
    with torch.no_grad():
        for inputs,labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # .cpu().numpy() cuz need to convert to pythorch tensor to a numpy array
            # while scikit-learn only knows numpy arrays, and numpy only knows cpu and not gpu (device)
            list_pred.extend(outputs.argmax(1).cpu().np())
            list_labels.extend(labels.np())

    return list_pred, list_labels

def save_confusion_matrix(model, loader, device, class_names, save_path):
    predictions, labels = get_predictions(model, loader, device)
    conf_matrix = confusion_matrix(labels, predictions)
    display = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)

    fix, ax = plt.subplots(figsize=(10, 8))
    display.plot(ax=ax, xticks_rotation=45, colorbar=True)
    plt.title("Confusion matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved at {save_path}")

def print_classification_report(model, loader, device, class_names):
    """
    Print per-class precision, recall and F1-score.
    Useful to detect which emotions are hardest to classify.
    """

    predictions, labels = get_predictions(model, loader, device)
    report = classification_report(labels, predictions, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    return report