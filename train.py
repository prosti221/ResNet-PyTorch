import torch
import argparse
from model import ResNet
from utils.data_loader import *
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

def get_args():
    parser = argparse.ArgumentParser(description='Training script for a AlexNet')
    parser.add_argument('-checkpoint', type=str, default='none', help='path to model checkpoint')
    parser.add_argument('-dataset', '--dataset', default='./data/ILSVRC2012_img_val', help='path to training data')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=90, help='number of epoch')
    parser.add_argument('--classes', type=int, default=1000, help='number of classes')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = args.batch_size
    lr = args.learning_rate
    num_epoch = args.epoch
    num_classes = args.classes 

    loss = 0.0
    epoch_checkpoint = 0
    steps = 1

    model = ResNet(num_classes, stack_depth=[2, 4, 6, 3]).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Load from checkpoint
    if args.checkpoint != 'none':
        try:
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_checkpoint = checkpoint['epoch']
            steps = checkpoint['steps']
            loss = checkpoint['loss']
        except FileNotFoundError:
            print("Error: Checkpoint file not found")
    model.to(device)


    # For CIFAR100
    train_loader, val_loader = cifar100_dataloader(args.dataset, args.batch_size, args.num_workers, download=True) 

    # For ImageNet 
    #train_loader = imageNet_dataloader(args.dataset, args.batch_size, args.num_workers)

    # For loading ImageNet into a train/validation split
    #train_loader = imageNet_dataloader_tv(args.dataset, args.batch_size, args.num_workers)

    writer = SummaryWriter()
    for epoch in range(epoch_checkpoint, num_epoch):
        running_loss = 0.0
        running_corrects = 0
        total_examples = 0
        model.train(True)
        
        train_correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

            writer.add_scalar('Training Loss', loss.item(), steps)
            
            
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_examples += labels.size(0)

            if steps % 100 == 0:
                with torch.no_grad():
                    _, preds = torch.max(outputs, 1)
                    accuracy = torch.sum(preds == labels)

                    writer.add_scalar('Training Accuracy', accuracy.item(), steps)

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                        .format(epoch + 1, steps, loss.item(), accuracy.item()))

            if steps % 1000 == 0:
                checkpoint_path = "./checkpoints/checkpoint_epoch{}_step{}.pth".format(epoch+1, steps)
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'steps': steps,
                        }, checkpoint_path)
                
            steps += 1
        #lr_scheduler.step()
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (running_corrects.double() / len(train_loader.dataset)) * 100.0
        
        if (epoch + 1) % 10 == 0:
            writer.add_graph(model, inputs)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        for inputs, labels in val_loader:
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
        val_acc = (val_correct / val_total) * 100

        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_acc, epoch)
        
        writer.add_scalar('Training Running Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_acc, epoch)

        print('\nEpoch: {} \tStep: {} \tTrain Loss: {:.4f} \tVal Loss: {:.4f} \tVal Acc: {:.4f}'
                .format(epoch + 1, steps, loss.item(), val_loss, val_acc))
        
        print('Epoch [{}/{}], Running loss: {:.4f}, Avg acc: {:.2f}\n'
          .format(epoch+1, num_epoch, epoch_loss, epoch_acc))
        
        writer.add_scalars('Accuracy', {'training': epoch_acc,
                                        'validation': val_acc}, epoch)
        if (epoch + 1) % 10 == 0:
            writer.add_graph(model, inputs)

    writer.close()
    torch.save({
            'epoch': num_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'steps': steps,
            }, './models/model.pt')
    print('Finished Training')
