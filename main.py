################################################################################
# Title:            main.py                                                    #
# Description:      Main script for training a simple MLP on MNIST.            #
# Author:           Aidin Attar                                                #
# Date:             2024-07-01                                                 #
# Version:          0.1                                                        #
# Usage:            python main.py                                             #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import os
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from simple_mlp import SimpleMLPTrainer
from simple_cnn import SimpleCNNTrainer

def do_save_func(epoch):
    return True

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a Neural Network and investigate the information plane.')
    parser.add_argument('--model', type=str, default='mlp', help='Model to train (simple_mlp or simple_cnn).')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function to use in the network.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the network.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda).')
    parser.add_argument('--mi_method', type=str, default='binning2', help='Method to use for estimating mutual information.')
    parser.add_argument('--verbose', action='store_true', help='Print training information.')
    args = parser.parse_args()
    
    # Obtain the plotter and generate plots
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    if not os.path.exists(f'./plots/{args.model}'):
        os.makedirs(f'./plots/{args.model}')
    if not os.path.exists(f'./plots/{args.model}/{args.activation}'):
        os.makedirs(f'./plots/{args.model}/{args.activation}')

    if not os.path.exists(f'./results/{args.model}'):
        os.makedirs(f'./results/{args.model}')
    if not os.path.exists(f'./results/{args.model}/{args.activation}'):
        os.makedirs(f'./results/{args.model}/{args.activation}')

    # Load dataset
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Train the model
    if args.model == 'mlp':
        trainer = SimpleMLPTrainer(train_loader, val_loader, args.activation, epochs=args.epochs, device=args.device, mi_method=args.mi_method, verbose=args.verbose, do_save_func=do_save_func, save_dir=f'./results/{args.model}/{args.activation}')
    elif args.model == 'cnn':
        trainer = SimpleCNNTrainer(train_loader, val_loader, args.activation, epochs=args.epochs, device=args.device, mi_method=args.mi_method, verbose=args.verbose, do_save_func=do_save_func, save_dir=f'./results/{args.model}/{args.activation}')
    else:
        raise ValueError(f'Invalid model: {args.model}')

    trainer.train()
    
    plotter = trainer.get_plotter()
    plotter.plot_information_plane(save_path=os.path.join(f'./plots/{args.model}', 'information_plane.png'))
    plotter.plot_loss_accuracy(save_path=os.path.join(f'./plots/{args.model}', 'loss_accuracy.png'))