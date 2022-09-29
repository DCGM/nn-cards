# file train.py
# author Michal Hradiš, Kristína Hostačná, Pavel Ševčík

import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG)
import json
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean
from time import time

import torch
import tqdm
from torch_geometric.loader import DataLoader

from src.dataset import GraphDataset
from src.graphbuilders import graph_builder_factory
from src.model import model_factory
from src.utils import Stats, json_str_or_path

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data-path", type=Path, help="Path to the csv data file", required=True)
    parser.add_argument("--model-config", type=json_str_or_path,
                        help="Json string or path to a json file containing model config.", required=True)
    parser.add_argument("--data-config", type=json_str_or_path,
                        help="Json string or path to a json file containing data config.", required=True)
    parser.add_argument("--start-iteration", default=0, type=int)
    parser.add_argument("--max-iterations", default=50000, type=int)
    parser.add_argument("--view-step", default=1000, type=int)
    parser.add_argument("-i", "--in-checkpoint", type=str)
    parser.add_argument("-o", "--out-checkpoint", type=str)
    parser.add_argument("--checkpoint-dir", default=Path("."), type=Path)
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.0002, help="Learning rate for ADAM.")
    parser.add_argument("--device", type=torch.device, help="The device to train on", default=torch.device("cuda"))

    args = parser.parse_args()
    return args

def dataloaders_factory(data_path, batch_size, graph_build_config):
    dataset = GraphDataset(data_path, graph_builder_factory(graph_build_config))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    return dataloader, None

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

def load_json_file(filename):
    with open(filename) as f:
        return json.load(f)

def log_progress(statistics: Stats, step: int, timespan):
        """Log the current progress

        Args:
            statistics: the current statistics
            step:       the current step
            timespan:   the time to log"""
        logging.info(f"Step: {step:d}, time: {timespan:.2f} s, " + ", ".join([f"{key:s}: {mean(variable):4.10f}" for key, variable in statistics.data.items()]))

def main():
    args = parse_arguments()
    logging.info("Initializing...")

    graph_build_config = load_json_file(args.graph_build_config)
    optimization_config = load_json_file(args.optimization_config)
    backbone_config = load_json_file(args.backbone_config)
    head_config = load_json_file(args.head_config)

    dataloader_train, dataloader_val = dataloaders_factory(args.data_path, args.batch_size, graph_build_config)

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

    logging.info("Training...")
    stats = Stats()
    train_iterator = continuous_iterator(dataloader_train)
    start_time = time()
    original_start_time = start_time
    for iteration in tqdm.tqdm(range(args.start_iteration, args.max_iterations), initial=args.start_iteration):
        batch = next(train_iterator)
        batch.to(args.device)

        optimizer.zero_grad()
        
        losses = model.compute_loss(batch)
        model.do_backward_pass(losses)
        optimizer.step()
        stats.add({key: value.item() for key, value in losses.items()})

        if iteration % args.view_step == 0:
            if args.out_checkpoint:
                checkpoint_path = args.out_checkpoint
            elif args.checkpoint_dir:
                args.checkpoint_dir.mkdir(exist_ok=True)
                checkpoint_path = args.checkpoint_dir / f"checkpoint_{iteration:06d}.pth"
            torch.save(model.state_dict(), checkpoint_path)

            eval_results = model.evaluate(dataloader_val)
            stats.add(eval_results)
            model.train()
            
            log_progress(stats, iteration, time() - start_time)
            start_time = time()
            stats.clear()

    logging.info(f"Finished training loop, total time: {(time() - original_start_time):.2f} s.")


if __name__ == "__main__":
    main()
