import argparse
import torch 
import json
import torch.nn as nn
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 
import os

from dataset import TESSDataset, RAVDESSDataset, TESS_EMOTIONS, RAVDESS_EMOTIONS
from models import get_model
from evaluate import evaluate, save_confusion_matrix, print_classification_report
from utils import set_seed

def get_dataset(args):
    # load the appropriate dataset 
    if args.dataset == 'TESS':
        return TESSDataset(args.data_path), TESS_EMOTIONS
    elif args.dataset == 'RAVDESS':
        return RAVDESSDataset(args.data_path), RAVDESS_EMOTIONS
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15):
    # Split dataset into train/val/test

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])

def save_training_curves(train_accs, val_accs, save_path):
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label='Train accuracy', marker='o')
    plt.plot(epochs, val_accs, label='Val accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Traning vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Traning curves saved at: {save_path}")

def experiments(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Model: {args.model}\n")
    print(f"Dataset: {args.dataset}\n")

    dataset, emotions = get_dataset(args)
    num_classes = len(emotions)
    print(f"Samples: {len(dataset)}")
    print(f"Classes: {num_classes} -> {emotions}\n")

    train_set, val_set, test_set = split_dataset(dataset)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=16, 
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size, 
        num_workers=16, 
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size, 
        num_workers=16, 
        pin_memory=True
    )

    model = get_model(args.model, num_classes, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    os.makedirs('results', exist_ok=True)
    best_val_acc = 0.0
    train_accs, val_accs = [], []
    best_model_path = f"results/{args.model}_{args.dataset}_best.pth"

    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_t = 0, 0, 0

        # train for each epoch 
        # tqdm for better train involving 
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_fn = criterion(outputs, labels)
            loss_fn.backward()
            optimizer.step()

            loss_t += loss_fn.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct/total
        val_acc = evaluate(model, val_loader, device) # validation accuration on val_set
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Loss: {loss_fn/len(train_loader):.4f} | "
            f"Train: {train_acc:.4f} | "
            f"Val: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path} (val_acc={best_val_acc:.4f})")

    
    # Test validation   
    print(f"\nLoading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    test_acc = evaluate(mod, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")

    # print the per-class report 
    print_classification_report(model, test_loader, device, emotions)

    #save the confusion matrix 
    save_confusion_matrix(model, test_loader, device, emotions, f"results/{args.model}_{args.dataset}_confusion.png")

    # save training curves 
    save_training_curves(train_accs, val_accs, f"results/{args.model}_{args.dataset}_curves.png")

    # save history in a .json
    history = {
        'model': args.model,
        'dataset': args.dataset, 
        'epochs': args.epochs, 
        'lr': args.lr, 
        'batch_size': args.batch_size, 
        'train_acc': train_accs, 
        'val_acc': val_accs,  
        'test_acc': test_acc, 
    }

    history_path = f"results/{args.model}_{args.dataset}_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

    print(f"History saved at {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition - Transfer Learning')
    parser.add_argument(
        "--model", type=str, default="vgg16", 
        choices=["vgg16", "resnet18", "resnet34", "resnet50"], 
        help="Backbone architecture"
    )
    parser.add_argument(
        "--dataset", type=str, default="TESS",
        choices=["TESS", "RAVDESS"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--data-path", type=str, required=True, 
        help="Path to dataset root folder",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, 
        help="Training epcohs number"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, 
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, 
        help="Learning rate"
    )

    args = parser.parse_args()
    experiments(args)
