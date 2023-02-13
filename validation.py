import torch
import argparse
from tqdm import tqdm
from model import AlexNet
from utils.data_loader import *
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser(description='Validation script for a AlexNet')
    parser.add_argument('-model', type=str, default='./models/model.pth', help='path to model')
    parser.add_argument('-dataset', '--dataset', default='./data/ILSVRC2012_img_val', help='path to validation data')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading')
    parser.add_argument('--classes', type=int, default=1000, help='number of classes in the model')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # ImageNet
    #val_loader = imageNet_dataloader(args.dataset, args.batch_size, args.num_workers)

    #CIFAR-10
    _, val_loader = cifar100_dataloader(args.dataset, args.batch_size, args.num_workers, download=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    model = AlexNet(args.classes)

    # Load model
    try:
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        print("Error: Checkpoint file not found")

    model.to(device)
    model.eval()  

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    for inputs, labels in tqdm(val_loader, desc='Validation Progress'):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        val_loss += loss.item() * inputs.size(0)

        _, predictions = torch.max(outputs, 1)
        val_correct += (predictions == labels).sum().item()
        val_total += inputs.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

