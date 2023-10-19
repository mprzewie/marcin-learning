import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import matplotlib.pyplot as plt
import wandb

# import argparse

from resnet import ResNet18, ResNet
# from utils import progress_bar
from collections import defaultdict
from sklearn.decomposition import PCA
import seaborn as sns
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


NUM_CLASSES = 100
class DataSplits:
    base_classes = "base_classes"
    new_classes = "new_classes"
    all_classes = "all_classes"

class Actions:
    train = "train"
    eval_linear = "eval-linear"
    eval_collapse = "eval-collapse"
    visualize_activations = "vis-activations"

def get_dataloaders(
    cutoff_class: int,
    batch_size: int,
    num_workers: int
) -> Dict[str, Tuple[DataLoader, DataLoader]]:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)

    indices_train_1 = [i for i, (x, y) in enumerate(trainset) if y < cutoff_class]
    indices_train_2 = [i for i, (x, y) in enumerate(trainset) if y >= cutoff_class]

    indices_test_1 = [i for i, (x, y) in enumerate(testset) if y < cutoff_class]
    indices_test_2 = [i for i, (x, y) in enumerate(testset) if y >= cutoff_class]

    trainset_1, trainset_2 = [
        torch.utils.data.Subset(trainset, inds)
        for inds in [indices_train_1, indices_train_2]
    ]

    testset_1, testset_2 = [
        torch.utils.data.Subset(testset, inds)
        for inds in [indices_test_1, indices_test_2]
    ]

    trainloader_1, trainloader_2, trainloader_12 = [
        torch.utils.data.DataLoader(d, batch_size=128, shuffle=True, num_workers=2)
        for d in [trainset_1, trainset_2, trainset]
    ]

    testloader_1, testloader_2, testloader_12 = [
        torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for d in [testset_1, testset_2, testset]
    ]

    return {
        DataSplits.base_classes: (trainloader_1, testloader_1),
        DataSplits.new_classes: (trainloader_2, testloader_2),
        DataSplits.all_classes: (trainloader_12, testloader_12)
    }

def train_epoch(
    epoch: int,
    model: ResNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    entropy_lambda: float
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    train_loss = 0
    train_entropy_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(loader), desc=f"{epoch}: training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, return_activations=True)

        l3_representations = outputs["l3"].mean(axis=(2, 3))

        #########
        # TODO entropy experiments

        l3_softmax = torch.nn.functional.softmax(l3_representations, dim=1)
        neg_l3_entropy = l3_softmax * l3_softmax.log()
        # entropy would have a minus sign, but I want to maximize entropy, so I need to minimize negative-entropy
        entropy_loss = neg_l3_entropy.sum(dim=1).mean()

        #########

        logits = outputs["out"]
        loss = criterion(logits, targets)
        (loss + (entropy_lambda * entropy_loss)).backward()
        optimizer.step()

        train_loss += loss.item()
        train_entropy_loss += entropy_loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        "epoch": epoch,
        "loss/train": train_loss / (batch_idx + 1),
        "loss_entropy/train": train_entropy_loss / (batch_idx + 1),
        "accuracy/train": 100. * correct / total
    }

def test_epoch(
    epoch: int,
    model: ResNet,
    loader: DataLoader,
):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(loader), desc=f"{epoch}: testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return {
        "epoch": epoch,
        "accuracy/test": 100. * correct / total,
        "loss/test": test_loss / (batch_idx + 1),
    }


def train(
    model: ResNet,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    experiment_dir: Path,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    entropy_lambda: float,
) -> List[Dict[str, float]]:
    metrics_list = []
    optimizer = optim.SGD(
        model.parameters(), lr=lr,
        momentum=0.9, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_accuracy = 0
    for epoch in range(num_epochs):
        train_metrics = train_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            entropy_lambda=entropy_lambda
        )
        scheduler.step()
        test_metrics = test_epoch(
            epoch=epoch,
            model=model,
            loader=test_loader
        )


        if test_metrics["acc"] > best_accuracy:
            print(f"Best acc increased from {best_accuracy:.2f} to {test_metrics['acc']:.f}. Saving.")
            best_accuracy = test_metrics["acc"]
            state = {
                'net': model.state_dict(),
                'acc': best_accuracy,
                'epoch': epoch,
            }
            torch.save(state, experiment_dir / "ckpt.pth")

        test_metrics["best_accuracy"] = best_accuracy

        test_metrics.update(train_metrics)
        ### TODO load to wandb

        metrics_list.append(test_metrics)

    return metrics_list


def linear_eval(
        train_activations: torch.Tensor, train_targets: torch.Tensor,
        test_activations: torch.Tensor, test_targets: torch.Tensor,
        num_ws=5, w_min=-2, w_max=5
):
    def build_step(X, Y, classifier, optimizer, w, criterion_fn):
        def step():
            optimizer.zero_grad()
            loss = criterion_fn(classifier(X), Y, reduction='sum')
            for p in classifier.parameters():
                loss = loss + p.pow(2).sum().mul(w)
            loss.backward()
            return loss

        return step

    best_acc = 0

    _, dim1 = train_activations.shape
    _, dim2 = test_activations.shape
    assert dim1 == dim2

    for w in torch.logspace(w_min, w_max, steps=num_ws).tolist():
        cls = nn.Linear(dim1, NUM_CLASSES).to(device).train()
        optimizer = torch.optim.LBFGS(
            cls.parameters(),
            line_search_fn='strong_wolfe', max_iter=5000, lr=1, tolerance_grad=1e-10, tolerance_change=0,
        )

        optimizer.step(build_step(train_activations, train_targets, cls, optimizer, w,
                                  criterion_fn=torch.nn.functional.cross_entropy))
        cls.eval()
        y_test_pred = cls(test_activations).argmax(dim=1)
        acc = (y_test_pred == test_targets).float().mean().item()

        if acc > best_acc:
            best_acc = acc

    return best_acc

def get_detached_activations(model: ResNet, loader: DataLoader) -> Dict[str, torch.Tensor]:
    acts = defaultdict(list)
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, return_activations=True)
            for k, v in outputs.items():

                if len(v.shape) == 4:
                    v = v.mean(axis=(2, 3))
                acts[k].extend(v)
            acts["y_true"].extend(targets.detach())

        return {
            k: np.array(v)
            for (k,v) in acts.items()
        }

def collapse_eval(
    activations: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:

    result = dict()
    activations, targets = [t.detach().cpu().numpy() for t in [activations, targets]]

    pca = PCA().fit(activations)
    exp_var = pca.explained_variance_ratio_

    cum = np.cumsum(exp_var)
    for i, c in enumerate(cum):
        if c > 0.9:
            break

    result["all-classes"] = i

    per_class_results = []

    for cls in list(set(targets)):
        index = np.array(activations["y_true"]) == cls
        c_fts = activations[index]
        pca = PCA().fit(c_fts)
        exp_var = pca.explained_variance_ratio_

        cum = np.cumsum(exp_var)
        for i, c in enumerate(cum):
            if c > 0.9:
                break

        per_class_results.append(i)

    result["per-class"] = np.mean(per_class_results)

    return result

def activations_visualization(
    activations: torch.Tensor,
    targets: torch.Tensor,
    title: str
) -> np.ndarray:
    per_class_acts = []

    mn = activations.mean(axis=0)

    # print(mn.shape)

    for c in sorted(list(set(targets))):  # todo check all
        index = activations["y_true"] == c
        c_acts = activations[index]

        per_class_acts.append(
            c_acts.mean(axis=0) - mn
        )

    mp = np.array(per_class_acts)


    heatmap = sns.heatmap(mp)
    plt.ylabel("class")
    plt.xlabel("features")
    plt.title(title)

    return heatmap.get_figure()

def maybe_setup_wandb(logdir, args=None, run_name_suffix=None, **init_kwargs):

    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT")

    if wandb_entity is None or wandb_project is None:
        print(f"{wandb_entity=}", f"{wandb_project=}")
        print("Not initializing WANDB")
        return

    origin_run_name = Path(logdir).name

    api = wandb.Api()

    name_runs = list(api.runs(f'{wandb_entity}/{wandb_project}', filters={'display_name': origin_run_name}))
    group_runs = list(api.runs(f'{wandb_entity}/{wandb_project}', filters={'group': origin_run_name}))

    print(f'Retrieved {len(name_runs)} for run_name: {origin_run_name}')

    assert len(name_runs) <= 1, f'retrieved_runs: {len(name_runs)}'

    new_run_name = origin_run_name if len(name_runs) == 0 else f"{origin_run_name}_{len(group_runs)}"

    if run_name_suffix is not None:
        new_run_name = f"{new_run_name}_{run_name_suffix}"

    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        config=args,
        name=new_run_name,
        dir=logdir,
        resume="never",
        group=origin_run_name,
        **init_kwargs
    )

    print("WANDB run", wandb.run.id, new_run_name, origin_run_name)

def main(args):

    run_name = f"resnet18_cc{args.cutoff_class}_bs{args.batch_size}_dr{str(args.detach_residual)[0]}_lr{args.lr}_wd{args.weight_decay}_elbd{args.ent_lambda}"

    experiment_dir = args.save_dir / run_name

    maybe_setup_wandb(logdir=experiment_dir, args=args)

    dataloaders = get_dataloaders(
        cutoff_class=args.cutoff_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    net = ResNet18(num_classes=NUM_CLASSES, detach_residual=args.detach_residual)

    # TODO - loading / restarting experiment logic
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if Actions.train in args.actions:
        train_metrics = train(
            model=net,
            train_loader=dataloaders[DataSplits.base_classes][0],
            test_loader=dataloaders[DataSplits.base_classes][1],
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            entropy_lambda=args.ent_lambda
        )


    checkpoint = torch.load(experiment_dir / "ckpt.pth")
    net.load_state_dict(checkpoint["net"], strict=True)
    net.eval()

    analysis_metrics = dict()

    for set_name, (trainloader, testloader) in dataloaders.items():

        train_activations = get_detached_activations(model=net, loader=trainloader)
        test_activations = get_detached_activations(model=net, loader=testloader)

        for acts in [train_activations, test_activations]:
            print({
                k: v.shape
                for (k,v) in acts.items()
            })

        if Actions.eval_linear in args.actions:
            for block in ["l4", "l3"]:
                block_accuracy = linear_eval(
                    train_activations=train_activations[block],
                    train_targets=train_activations["y_true"],
                    test_activations=test_activations[block],
                    test_targets=test_activations["y_true"]
                )
                analysis_metrics[f"linear_eval/{set_name}/{block}"] = block_accuracy

        if Actions.eval_collapse in args.actions:
            for block in ["l1", "l2", "l3", "l4"]:
                block_collapse_metrics = collapse_eval(
                    activations=test_activations[block],
                    targets=test_activations["y_true"]
                )
                for k, v in block_collapse_metrics.items():
                    analysis_metrics[f"collapse/{k}/{set_name}/{block}"] = v

        if Actions.visualize_activations in args.actions:
            for block in ["l1", "l2", "l3", "l4"]:
                ttl = f"activations/{set_name}/{block}"
                visualization = activations_visualization(
                    activations=test_activations[block],
                    targets=test_activations["y_true"],
                    title=ttl
                )
                analysis_metrics[ttl] = wandb.Image(
                    wandb.Image(
                        visualization,
                        caption=ttl
                    )
                )









if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--save-dir", type=Path, required=True)
    # parser.add_argument("--experiment-dir", type=Path, required=True )

    parser.add_argument(
        "--actions", nargs='+', type=str,
        choices=[
            Actions.train,
            Actions.eval_linear,
            Actions.eval_collapse,
            Actions.visualize_activations
        ]
    )

    parser.add_argument("--cutoff-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--detach-residual", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--ent-lambda", type=float, default=0)

    parser.add_argument()
    args = parser.parse_args()

    main(args)



