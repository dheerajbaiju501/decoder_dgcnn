from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import PointNet, DGCNN, Decoder
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import pdb


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints/' + args.exp_name + '/main.py.backup')
    os.system('cp model.py checkpoints/' + args.exp_name + '/model.py.backup')
    os.system('cp util.py checkpoints/' + args.exp_name + '/util.py.backup')
    os.system('cp data.py checkpoints/' + args.exp_name + '/data.py.backup')


def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'pointnet':
        encoder = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        encoder = DGCNN(args).to(device)
    else:
        raise Exception("Model not implemented")
    print(str(encoder))

    decoder = Decoder(latent_dim=args.emb_dims, grid_size=45).to(device)

    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

    if args.use_sgd:
        opt = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), 
                        lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                         lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        train_loss = 0.0
        count = 0.0
        encoder.train()
        decoder.train()
        train_pred = []
        train_true = []
        for batch in train_loader:
            data = batch[0]
            label = batch[1]
            data, label = data.to(device), label.to(device).squeeze()
            

        
            batch_size = data.size()[0]
            opt.zero_grad()

            latent_vector = encoder(data)
            reconstructed_points = decoder(latent_vector)
            loss = criterion(reconstructed_points, data)

            loss.backward()
            opt.step()

            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())

        train_true = np.concatenate(train_true)
        outstr = 'Train %d, loss: %.6f' % (epoch, train_loss * 1.0 / count)
        io.cprint(outstr)

        test_loss = 0.0
        count = 0.0
        encoder.eval()
        decoder.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            latent_vector = encoder(data)
            reconstructed_points = decoder(latent_vector)

            loss = criterion(reconstructed_points, data)
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())

        test_true = np.concatenate(test_true)
        outstr = 'Test %d, loss: %.6f' % (epoch, test_loss * 1.0 / count)
        io.cprint(outstr)

        if test_loss <= best_test_acc:
            best_test_acc = test_loss
            torch.save(encoder.state_dict(), 'checkpoints/%s/models/encoder_model.t7' % args.exp_name)
            torch.save(decoder.state_dict(), 'checkpoints/%s/models/decoder_model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    encoder = DGCNN(args).to(device)
    decoder = Decoder(latent_dim=args.emb_dims, grid_size=45).to(device)

    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

    #pdb.set_trace()

     
    #encoder.load_state_dict(torch.load(args.model_path))  
    #decoder.load_state_dict(torch.load(args.model_path.replace("encoder", "decoder")))  # Same for decoder

    # ... existing code ...
    encoder.eval()
    decoder.eval()

    test_loss = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    
    for batch in test_loader:
        data = batch[0]
        label = batch[1]
        data, label = data.to(device), label.to(device).squeeze()
        if data.dim() == 4:
            # Assuming the shape is (batch_size, channels, height, width)
            data = data.permute(0, 2, 3, 1)
        elif data.dim() == 3:
            # If it's actually 3D, use the original permutation
            data = data.permute(0, 2, 1)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        batch_size = data.size()[0]
        oct.zero_grad()

        latent_vector = encoder(data)
        reconstructed_points = decoder(latent_vector)

        loss = criterion(reconstructed_points, data)
        count += batch_size
        test_loss += loss.item() * batch_size
        test_true.append(label.cpu().numpy())

    test_true = np.concatenate(test_true)
    outstr = 'Test :: test loss: %.6f' % (test_loss * 1.0 / count)
    io.cprint(outstr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Autoencoder')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint('Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
