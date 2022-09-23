# author: Pavel Ševčík

import logging
import json
from argparse import ArgumentParser
from pathlib import Path

import torch
import tqdm

from src.dataset import GraphDataset
from src.model import model_factory
from src.utils import Stats

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data-path", type=Path, help="Path to the csv data file", required=True)
    parser.add_argument("--graph-build-config", help="Json graph build config.", required=True)
    parser.add_argument("--backbone-config", help="Json network config.", required=True)
    parser.add_argument("--head-config", help="Json head config.", required=True)
    parser.add_argument("--optimization-config", help="Json optimization config", required=True)
    parser.add_argument("--device", type=torch.device, help="The device to train on", default=torch.device("cuda"))

    args = parser.parse_args()
    return args

def dataloaders_factory(data_path, graph_build_config):
    dataset = GraphDataset(data_path, graph_build_config)
    return dataset, None

def continuous_iterator(iterable):
    """Returns continuous iterator
    Args:
        iterable - iterable object to be iterated"""
    iterator = iter(iterable)
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            iterator = iter(iterable)
            item = next(iterator)
        yield item

def optimizer_factory(model, optimization_config):
    optimizer_type = optimization_config["type"].lower()
    del optimization_config["type"]
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimization_config)
    else:
        msg = f"Unknown optimizer type value '{optimizer_type}'"
        logging.error(msg)
        raise ValueError(msg)
    return optimizer

def evaluate(dataloader_val, model):
    model.eval()

def main():
    args = parse_arguments()

    graph_build_config = json.load(args.graph_build_config)
    optimization_config = json.load(args.optimization_config)
    backbone_config = json.load(args.backbone_config)
    head_config = json.load(args.head_config)

    dataloader_train, dataloader_val = dataloaders_factory(args.data_path, graph_build_config)

    model = model_factory(backbone_config, head_config)

    checkpoint_path = None
    if args.in_checkpoint:
        checkpoint_path = args.in_checkpoints
    elif args.start_iteration:
        checkpoint_path = args.checkpoint_dir / f"checkpoint_{args.start_iteration:06d}.pth"
    
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))

    model.train()
    model.to(args.device)

    optimizer = optimizer_factory(model, optimization_config)

    stats = Stats()
    train_iterator = continuous_iterator(dataloader_train)
    for iteration in tqdm.tqdm(args.start_iteration, args.max_iterations, initial=args.start_iteration):
        batch = next(train_iterator)
        batch.to(args.device)

        optimizer.zero_grad()
        
        losses = model.compute_loss(batch)
        model.do_backward_pass(losses)
        stats.add({key: value.item() for key, value in losses.items()})

        if iteration % args.view_step == 0:
            if args.out_checkpoint:
                checkpoint_path = args.out_checkpoint
            elif args.checkpoint_dir:
                args.checkpoint_dir.mkdir(exist_ok=True)
                checkpoint_path = args.checkpoint_dir / f"checkpoint_{iteration:06d}.pth"
            torch.save(model.state_dict(), checkpoint_path)

            model.evaluate(dataloader_val)
            stats.clear()


if __name__ == "__main__":
    main()
