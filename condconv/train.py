import argparse
from collections import defaultdict
from datetime import datetime
from math import ceil
from pathlib import Path

import numpy as np
import torch
# import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from models.resnet import ResNet18
# from models.resnet_condconv2d import ResNet18 as CondResNet18
#
# from models import resnet as models
# from models import resnet_condconv2d as cond_models
from cond_resnet import cond_resnet18, MAIN_FC_KS, FC_FOR_CHANNELS
from resnet import resnet18
# from resnet_regular_lukasz import ResNet18 as LukaszResnet18

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--epochs', default=25, type=int, help='number of epochs')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--data_root', default='./data', help='Directory where results will be save')
    parser.add_argument('--save_dir', default='./results', help='Directory where results will be save')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument("--model_type", default="classic", choices=["classic", "conditional"])
    parser.add_argument("--inplanes", default=64, type=int, help="ResNet inplanes")
    parser.add_argument('--conditional', '-c', nargs='+', type=float, default=None,
                        help="Train separate ResNet heads for specific numbers of channels."
                             "If not specified, one robust head will be trained")
    parser.add_argument('--k', default=None, type=int, help='Number of condition samples for one robust head.')
    parser.add_argument('--scheduler', choices=['None', 'CyclicLR', 'LambdaLR'], default='CyclicLR')
    parser.add_argument('--model', choices=[
        'ResNet18',
        # 'ResNet34', 'ResNet50'
        ], default='ResNet18')

    args = parser.parse_args()
    if args.conditional is not None:
        assert np.prod(args.conditional) > 0
        args.conditional.sort()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)

    # Model
    print('==> Building model..')

    model_str = f"{args.model_type}_{args.model}-inplanes_{args.inplanes}"
    if args.model_type == "classic":
        assert args.conditional is None and args.k is None
        net = resnet18(in_planes=args.inplanes, num_classes=10)
    else:
        if args.conditional is not None:
            assert args.k is None
            if any([i < 1 for i in args.conditional]):
                args.conditional = [ceil(args.inplanes * i) for i in args.conditional]
            else:
                args.conditional = [int(i) for i in args.conditional]

            model_str = f"{model_str}-{len(args.conditional)}_heads"
            net = cond_resnet18(in_planes=args.inplanes, fc_for_channels=args.conditional)

        else:
            assert args.k is not None
            model_str = f"{model_str}-robust_head_{args.k}_samples"
            net = cond_resnet18(in_planes=args.inplanes)

    print(f'\033[0;1;33m{model_str}\033[0m')

    args.save_dir = f"{args.save_dir}/{model_str}"

    net = net.to(device)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert Path(args.resume).is_file(), 'Error: resume file does not exist!'
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)

    if args.scheduler == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1.2 * args.lr, step_size_up=10,
                                                      mode="exp_range", scale_mode='cycle', cycle_momentum=False,
                                                      gamma=(1e-4 / args.lr) ** (1 / (0.9 * args.epochs)))
    elif args.scheduler == 'LambdaLR':
        gamma = (1e-4 / args.lr) ** (1 / (0.9 * args.epochs))
        labda = lambda epoch: gamma ** epoch if epoch < 0.9 * args.epochs else 1e-4 / args.lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=labda)
    else:
        scheduler = None

    args.save_dir = f"{args.save_dir}/{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    args.path_checkpoint = f'{args.save_dir}/checkpoint'
    args.path_logs = f'{args.save_dir}/logs'

    print('\033[0;32m', '=' * 15, '   Parameters   ', '=' * 15, '\033[0m', sep='')
    for arg in vars(args):
        print(f'\033[0;32m{arg}: {getattr(args, arg)}\033[0m')
    print('\033[0;32m', '=' * 46, '\033[0m', sep='')

    Path(args.path_checkpoint).mkdir(exist_ok=True, parents=True)

    with open(f"{args.save_dir}/config.yaml", 'w') as yaml_file:
        yaml.safe_dump(vars(args), yaml_file, default_style=None, default_flow_style=False, sort_keys=False)

    writer = SummaryWriter(args.path_logs)
    epoch_tqdm = tqdm(range(start_epoch, start_epoch + args.epochs), desc="Training")
    for epoch in epoch_tqdm:
        ############
        # Training #
        ############
        net.train()

        total = correct = train_loss = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            if args.model_type in ["classic"]:

                outputs = net(inputs)
                loss = criterion(outputs, targets)

            else:
                if args.conditional is None:
                    samples = np.arange(1, args.inplanes)
                    np.random.shuffle(samples)
                    samples = samples[:args.k]
                    samples = list(samples) + [args.inplanes]
                    loss = 0

                    _, inter = net(inputs, return_intermediate=True, main_fc_ks=samples)
                    for k, outputs in inter[MAIN_FC_KS].items():
                        loss += criterion(outputs, targets)

                else:
                    _, inter = net(inputs, return_intermediate=True)
                    loss = 0
                    for k, outputs in inter[FC_FOR_CHANNELS].items():
                        loss += criterion(outputs, targets)

            optimizer.zero_grad(True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # outputs = outputs if args.conditional is None else outputs[-1]
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

            total += targets.size(0)

            # ===================logger========================
            writer.add_scalar('train/loss', loss.item(), epoch * len(trainloader) + batch_idx)

        acc_train = correct / total
        train_loss /= len(trainloader)

        ############
        # Testing #
        ############
        net.eval()

        total = 0

        # correct = 0 if args.conditional is None else [0. for _ in args.conditional]
        # test_loss = 0 if args.conditional is None else [0. for _ in args.conditional]
        correct = defaultdict(float)
        test_loss = defaultdict(float)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), total=len(testloader), leave=False):
                inputs, targets = inputs.to(device), targets.to(device)

                if args.model_type in ["classic"]:
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    # ===================logger========================
                    writer.add_scalar(f'test_loss/{args.k}', loss.item(),
                                      epoch * len(testloader) + batch_idx)

                    test_loss[args.k] += loss.item()
                    _, predicted = outputs.max(1)
                    correct[args.k] += predicted.eq(targets).sum().item()

                else:
                    if args.conditional is None:
                        ks_to_check = sorted(set(list(range(1, args.inplanes, 8)) + [args.inplanes]))
                        _, inter = net(inputs, return_intermediate = True, main_fc_ks=ks_to_check)
                        out_key = MAIN_FC_KS
                    else:
                        _, inter = net(inputs, return_intermediate = True)
                        out_key = FC_FOR_CHANNELS

                    for k, outputs in inter[out_key].items():
                        loss = criterion(outputs, targets)
                        test_loss[k] += loss.item()
                        _, predicted = outputs.max(1)
                        correct[k] += predicted.eq(targets).sum().item()
                        writer.add_scalar(f'test_loss/{k}', loss.item(), epoch * len(testloader) + batch_idx)

                total += targets.size(0)

        # ===================logger========================
        writer.add_scalar('train/loss_per_epoch', train_loss, epoch)
        writer.add_scalar('train/acc_per_epoch', acc_train, epoch)


        for k, v in correct.items():
            acc_test = correct[k] / total
            test_loss[k] /= len(testloader)

            writer.add_scalar(f'test_loss_per_epoch/{k}', test_loss[k], epoch)
            writer.add_scalar(f'test_acc_per_epoch/{k}', acc_test, epoch)


        acc_test = correct[max(correct.keys())] / total
        test_loss = sum(test_loss.values())
        ############################
        #         Save model       #
        ############################
        if acc_test > best_acc:
            state = {'net': net.state_dict(), 'acc': acc_test, 'epoch': epoch, }
            torch.save(state, f'{args.save_dir}/checkpoint/model.pth')
            best_acc = acc_test

        ############################
        #         scheduler        #
        ############################
        if scheduler is None:
            writer.add_scalar('scheduler', args.lr, epoch)
        else:
            scheduler.step()
            writer.add_scalar('scheduler', scheduler.get_last_lr()[0], epoch)

        epoch_tqdm.set_description(f"Train: loss={train_loss:.4f}, acc={acc_train:.4f}, "
                                   f"Test: loss={test_loss:.4f}, acc={acc_test:.4f}")


if __name__ == '__main__':
    main()