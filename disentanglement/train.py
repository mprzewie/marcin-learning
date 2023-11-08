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
import random

from dataclasses import dataclass

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
    visaualize_feature_activations = "vis-feat-acts"
    visualize_kld = "vis-kld"

class EntropyOps:
    qsuare = "qsuare"
    add_min = "add-min"
    softmax = "softmax"
    normalize = "normalize"
    

def mean_class_activations(
    activations,
    targets,
    normalization=None,
    temperature=1
):
    oh_targets = torch.zeros(len(targets), targets.max()+1).to(activations.device)
    oh_targets[torch.arange(len(targets)), targets] = 1
    oh_targets = torch.nn.functional.normalize(oh_targets, p=1, dim=0)

    mean_activations = oh_targets.T @ activations
    
    if normalization == EntropyOps.softmax:
        mean_activations = torch.softmax(mean_activations / temperature, dim=1)
    elif normalization == EntropyOps.normalize:
        mean_activations = mean_activations ** temperature
        a_sum = mean_activations.sum(dim=1)
        mean_activations = (mean_activations.T / a_sum ).T
    
    
    return mean_activations
    

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


def entropy_losses(
    block_representations: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    temperature: float,
    ops: List[str],
) -> Dict[str, torch.Tensor]:
    
    assert (("softmax" in ops) != ("normalize" in ops)), ops
    
    result = dict()
    
    relevant_blocks = ["l1", "l2", "l3", "l4"]
    block_representations = {
        k: block_representations[k].mean(axis=(2,3))
        for k in relevant_blocks
    }
    
    for block, activations in block_representations.items():
        for axis in [0, 1]:
            for op in ops:
                if op == EntropyOps.qsuare:
                    activations = activations ** 2
                if op == EntropyOps.add_min:
                    min_act, _ = activations.min(dim=1)
                    activations = (activations - min_act.unsqueeze(dim=1)) + 1e-6
                    
                if op == EntropyOps.softmax:
                    activations = torch.nn.functional.softmax(
                        activations / temperature, 
                        dim=axis
                    )
                if op == EntropyOps.normalize:
                    activations = activations ** temperature
                    a_sum = activations.sum(dim=axis)
                    if axis == 0:
                        activations = activations / a_sum
                    elif axis == 1:
                        activations = (activations.T / a_sum ).T
                    else:
                        assert False, axis
                    
            assert 0 <= activations.min() <= activations.max() <= 1, (activations.min(), activations.max())
                
            neg_b_repr_entropy = activations * activations.log()
            b_ent = neg_b_repr_entropy.sum(dim=axis).mean()
            result[f"{axis}/{block}"] = b_ent
    
    return result


def inter_class_kld(
    activations: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    normalization: str,
    epsilon: float = 1e-6,
    drop_missing: bool = False
):
    per_class_acts = mean_class_activations(
        activations, targets,
        normalization=normalization,
        temperature=temperature,
    )
    if drop_missing:
        per_class_acts = per_class_acts[
            sorted(set(targets.detach().cpu().numpy()))
        ]
    
    per_class_acts = per_class_acts - per_class_acts.min()
    
    # assert not (torch.any(torch.isnan(per_class_acts)))
    
    b, f = per_class_acts.shape
    
    logdivs = torch.einsum(
        "np,mp->nmp",
        [
            (per_class_acts + epsilon), 
            1/(per_class_acts + epsilon)
        ]
    ).log()
    
    r = (per_class_acts.reshape(b, 1, f) * logdivs).sum(dim=2)
    
#     assert not (torch.any(torch.isnan(per_class_acts))), "pca"
#     assert not (torch.any(torch.isnan(logdivs))), "logdivs"

#     assert not (torch.any(torch.isnan(r))), (
#         "result", 
#         (per_class_acts.min(), per_class_acts.max()),
#         (logdivs.min(), logdivs.max()),
#         (r.min(), r.max())
#     )
        
    return r
    
    
        
        

def train_epoch(
    epoch: int,
    model: ResNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    entropy_temperature: float,
    entropy_lambda: float,
    ent_blocks: List[str],
    ent_axis: int,
    ent_ops: List[str],
    kld_lambda: float,
    kld_share: float,
    kld_norm: str,
    kld_blocks: List[str],
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    train_loss = 0
    correct = 0
    total = 0
    train_entropy_losses = defaultdict(float)
    train_kld_losses = defaultdict(float)
    
    effective_loss = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(loader), desc=f"{epoch}: training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, return_activations=True)
        logits = outputs["out"]
        loss = criterion(logits, targets)
        loss_and_entropy = loss

        ########
        ent_losses = entropy_losses(outputs, targets, temperature=entropy_temperature, ops=ent_ops)
        for k, v in ent_losses.items():
            train_entropy_losses[f"entropy/train/{k}"] += v.item()
        
        for b in ent_blocks:
            ent_loss_key = f"{ent_axis}/{b}"
            loss_and_entropy = loss_and_entropy + (entropy_lambda * ent_losses[ent_loss_key])
        

        #########
        
        kld_losses = dict()
        
        for b in ["l1", "l2", "l3", "l4"]:
            ikld = inter_class_kld(
                outputs[b].mean(axis=(2,3)), 
                targets=targets, drop_missing=True, normalization=kld_norm, temperature=1.0
            )
            ikld_sorted = sorted(ikld.detach().cpu().numpy().reshape(-1))
            thresh_elem = int((len(ikld_sorted) -1) * kld_share)
            thresh = ikld_sorted[thresh_elem]
            ikld = ikld * (ikld < thresh)
            kld_losses[f"kld/train/{b}"] = ikld.sum().item()
            kld_losses[f"kld/thresh/{b}"] = thresh
            
            if kld_lambda != 0:
                if b in kld_blocks:
                    loss_and_entropy = loss_and_entropy + (kld_lambda * ikld.sum())
            
        for k, v in kld_losses.items():
            train_kld_losses[k] += v
        
        #########

        
        
        (loss_and_entropy).backward()
        optimizer.step()
        
        effective_loss += loss_and_entropy.item()
        train_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        "epoch": epoch,
        "loss/train": train_loss / (batch_idx + 1),
        "loss/effective": effective_loss / (batch_idx + 1),
        "accuracy/train": 100. * correct / total,
        **{
            k: (v  / (batch_idx + 1))
            for (k,v) in train_entropy_losses.items()
        },
        **{
            k: (v  / (batch_idx + 1))
            for (k,v) in train_kld_losses.items()
        },
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
    entropy_temperature: float,
    entropy_lambda: float,
    ent_blocks: List[str],
    ent_axis: int,
    ent_ops: List[str],
    kld_lambda: float,
    kld_share: float,
    kld_norm: str,
    kld_blocks: List[str],
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
            entropy_lambda=entropy_lambda,
            entropy_temperature=entropy_temperature,
            ent_blocks=ent_blocks,
            ent_axis=ent_axis,
            ent_ops=ent_ops,
            kld_lambda=kld_lambda,
            kld_share=kld_share,
            kld_norm=kld_norm,
            kld_blocks=kld_blocks
            
        )
        scheduler.step()
        test_metrics = test_epoch(
            epoch=epoch,
            model=model,
            loader=test_loader
        )


        if test_metrics["accuracy/test"] > best_accuracy:
            print(f"Best acc increased from {best_accuracy:.2f} to {test_metrics['accuracy/test']:.2f}. Saving.")
            best_accuracy = test_metrics["accuracy/test"]
            state = {
                'net': model.state_dict(),
                'acc': best_accuracy,
                'epoch': epoch,
            }
            torch.save(state, experiment_dir / "ckpt.pth")

        test_metrics["accuracy/best"] = best_accuracy

        test_metrics.update(train_metrics)

        if wandb.run is not None:
            wandb.log(test_metrics)
        print(epoch, " | ".join(sorted([f"{k}: {v:.2f}" for (k, v) in test_metrics.items()])))

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
            k: torch.stack(v)
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
        index = targets == cls
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

    for c in sorted(list(set(targets.detach().cpu().numpy().tolist()))):
        index = targets == c
        c_acts = activations[index]
        
        # assert False, (c, c_acts.shape,  (c_acts.mean(axis=0) - mn).shape)

        per_class_acts.append(
            c_acts.mean(axis=0) - mn
        )
    
    mp = torch.stack(per_class_acts).detach().cpu().numpy()
    
    # assert False, (len(per_class_acts), mp.shape)

    plt.figure()
    heatmap = sns.heatmap(mp)
    plt.ylabel("class")
    plt.xlabel("features")
    plt.title(title)

    return heatmap.get_figure()

def activations_sorted_for_each_feature_visualization(
    activations: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    normalization: str,
    title: str,
) -> np.ndarray:
    
    per_class_acts = mean_class_activations(
        activations, targets, 
        normalization=normalization,
        temperature=temperature,
    )
    
    
    mp = per_class_acts.detach().cpu().numpy().T
    
    mp = np.array([
        sorted(r, reverse=True)
        for r in mp
    ])
    
    sorted_rows = np.array(sorted(np.arange(len(mp)), key=lambda i: mp[i].sum(), reverse=True)).astype(int)
    
    
    mp = mp[sorted_rows].T

    plt.figure()
    heatmap = sns.heatmap(mp)
    plt.ylabel("classes (sorted individually in\neach column by activation strength)")
    plt.xlabel("features (sorted by sum of mean activations in classes)")
    plt.title(title, fontsize="small")
    locs, labels = plt.xticks()
    
    plt.xticks(locs, sorted_rows[[int(l) for l in locs]])
    return heatmap.get_figure()


def kld_heatmap_visualization(
    activations: torch.Tensor,
    targets: torch.Tensor,
    title: str,
    temperature: float,
    normalization: str,
) -> np.ndarray:
    
    with torch.no_grad():
        kld = inter_class_kld(activations, targets, temperature=temperature, normalization=normalization).cpu().numpy()
    
    
    plt.figure()
    heatmap = sns.heatmap(kld)
    plt.ylabel("P in DKL(P || Q)")
    plt.xlabel("Q in DKL(P || Q)")
    plt.title(title, fontsize="small")
    
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
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        random.seed(args.seed)

    run_name = f"resnet18_e{args.epochs}_cc{args.cutoff_class}_bs{args.batch_size}_dr{str(args.detach_residual)[0]}_cl{args.cls_len}_lr{args.lr}_wd{args.weight_decay}_d{args.dropout}_e-lbd{args.ent_lambda}_t{args.ent_temperature}_b-{'-'.join(args.ent_blocks)}_o-{'-'.join([o[0] for o in args.ent_ops])}_a{args.ent_axis}_kl{args.kld_lambda}_ks{args.kld_share}_kn{str(args.kld_norm)[0]}_kb{'-'.join(args.kld_blocks)}_s{args.seed}"
    
    if args.suffix is not None:
        run_name += f"_{args.suffix}"
        
    experiment_dir = args.save_dir / run_name
    
    if Actions.train in args.actions:
        if (experiment_dir / "ckpt.pth").exists():
            print("warning - checkpoint exists, I will overwrite it!")
        experiment_dir.mkdir(parents=True, exist_ok=True)
    else:
        assert experiment_dir.exists(), experiment_dir

    maybe_setup_wandb(
        logdir=experiment_dir, args=args
    )

    dataloaders = get_dataloaders(
        cutoff_class=args.cutoff_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    net = ResNet18(num_classes=NUM_CLASSES, detach_residual=args.detach_residual, dropout=args.dropout, cls_len=args.cls_len)

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
            experiment_dir=experiment_dir,
            entropy_temperature=args.ent_temperature,
            entropy_lambda=args.ent_lambda,
            ent_blocks=args.ent_blocks,
            ent_axis=args.ent_axis,
            ent_ops=args.ent_ops,
            kld_lambda=args.kld_lambda,
            kld_share=args.kld_share,
            kld_norm=args.kld_norm,
            kld_blocks=args.kld_blocks
            
        )


    checkpoint = torch.load(experiment_dir / "ckpt.pth")
    net.load_state_dict(checkpoint["net"], strict=True)
    net.eval()

    for set_name, (trainloader, testloader) in dataloaders.items():

        train_activations = get_detached_activations(model=net, loader=trainloader)
        test_activations = get_detached_activations(model=net, loader=testloader)

        
        for i, block in enumerate(
            ["l1", "l2", "l3", "l4"]
        ):
            analysis_metrics = dict(
                resnet_block=i+1
            )

            
            if Actions.eval_linear in args.actions:
                block_accuracy = linear_eval(
                    train_activations=train_activations[block],
                    train_targets=train_activations["y_true"],
                    test_activations=test_activations[block],
                    test_targets=test_activations["y_true"]
                )
                analysis_metrics[f"linear_eval/{set_name}"] = block_accuracy

            if Actions.eval_collapse in args.actions:
                block_collapse_metrics = collapse_eval(
                    activations=test_activations[block],
                    targets=test_activations["y_true"]
                )
                for k, v in block_collapse_metrics.items():
                    analysis_metrics[f"collapse/{k}/{set_name}"] = v

            if Actions.visualize_activations in args.actions:
                ttl = f"activations/{set_name}"
                visualization = activations_visualization(
                    activations=test_activations[block],
                    targets=test_activations["y_true"],
                    title=f"{run_name}\n{ttl}/{block}"
                )
                analysis_metrics[ttl] = wandb.Image(
                    wandb.Image(
                        visualization,
                        caption=f"{ttl}/{block}"
                    )
                )
            if Actions.visaualize_feature_activations in args.actions:
                for normalization in [None, EntropyOps.softmax, EntropyOps.normalize]:
                    ttl = f"activations_sorted_for_features/{normalization}/{set_name}"
                    visualization = activations_sorted_for_each_feature_visualization(
                        activations=test_activations[block],
                        targets=test_activations["y_true"],
                        title=f"{run_name}\n{ttl}/{block}",
                        temperature=args.ent_temperature,
                        normalization=normalization,
                    )
                    analysis_metrics[ttl] = wandb.Image(
                        wandb.Image(
                            visualization,
                            caption=f"{ttl}/{block}"
                        )
                    )
                
            if Actions.visualize_kld in args.actions:
                for normalization in [None, EntropyOps.softmax, EntropyOps.normalize]:
                    ttl = f"inter_class_kld/{normalization}/{set_name}"
                    visualization = kld_heatmap_visualization(
                        activations=test_activations[block],
                        targets=test_activations["y_true"],
                        title=f"{run_name}\n{ttl}/{block} ({args.ent_temperature:.2f})",
                        temperature=args.ent_temperature,
                        normalization=normalization,
                    )
                    analysis_metrics[ttl] = wandb.Image(
                        wandb.Image(
                            visualization,
                            caption=f"{ttl}/{block}"
                        )
                    )

            if wandb.run is not None:
                wandb.log(analysis_metrics)



if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--suffix", type=str)

    parser.add_argument(
        "--actions", nargs='+', type=str,
        choices=[
            Actions.train,
            Actions.eval_linear,
            Actions.eval_collapse,
            Actions.visualize_activations,
            Actions.visaualize_feature_activations,
            Actions.visualize_kld,
        ],
        default=[
            Actions.train,
            Actions.eval_linear,
            Actions.eval_collapse,
            Actions.visualize_activations,
            Actions.visaualize_feature_activations,
            Actions.visualize_kld,
        ]
    )

    parser.add_argument("--cutoff-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--detach-residual", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--cls-len", type=int, default=1)
    
    parser.add_argument("--ent-lambda", type=float, default=0.0)
    parser.add_argument("--ent-temperature", type=float, default=1)
    parser.add_argument(
        "--ent-blocks", nargs="+", type=str, 
        choices=["l1", "l2", "l3", "l4"], 
        default=["l3"]
    )
    parser.add_argument(
        "--ent-ops", nargs="+", 
        choices=[EntropyOps.qsuare, EntropyOps.add_min, EntropyOps.softmax, EntropyOps.normalize],
        default=[EntropyOps.softmax]
    )
    parser.add_argument("--ent-axis", choices=[0,1], default=1, type=int)
    
    parser.add_argument("--kld-lambda", type=float, default=0.0)
    parser.add_argument("--kld-share", type=float, default=1.0)
    parser.add_argument("--kld-norm", type=str, choices=[EntropyOps.softmax, EntropyOps.normalize, None], default=None)
    
    parser.add_argument("--kld-blocks", nargs="+", type=str, 
        choices=["l1", "l2", "l3", "l4"], 
        default=["l3"]
    )
    
    
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()

    main(args)



