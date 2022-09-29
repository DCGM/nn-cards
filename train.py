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

from src.utils import Stats, json_str_or_path
from src.config import Config

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data-path", type=Path, help="Path to the csv data file", required=True)
    parser.add_argument("--model-config", type=json_str_or_path,
                        help="Json string or path to a json file containing model config.", required=True)
    parser.add_argument("--data-config", type=json_str_or_path,
                        help="Json string or path to a json file containing data config.", required=True)
    parser.add_argument("--opt-config", type=json_str_or_path,
                        help="Json string or path to a json file containing optimization config.", required=True)
    parser.add_argument("--start-iteration", default=0, type=int)
    parser.add_argument("--max-iterations", default=50000, type=int)
    parser.add_argument("--view-step", default=1000, type=int)
    parser.add_argument("-i", "--in-checkpoint", type=str)
    parser.add_argument("-o", "--out-checkpoint", type=str)
    parser.add_argument("--checkpoint-dir", default=Path("."), type=Path)
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.0002, help="Learning rate for ADAM.")
    parser.add_argument("--device", type=torch.device, help="The device to train on", default=torch.device("cuda"))
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    args = parser.parse_args()
    return args

def dataloader_factory(dataset, batch_size, num_workers):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    return dataloader

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

    config = Config()
    dataset_factory, model = config.configure(args.data_config, args.model_config)
    dataset_train = dataset_factory(args.data_path)

    dataloader_train = dataloader_factory(dataset_train, args.batch_size, args.dataloader_num_workers)
    dataloader_val = None # TODO

    checkpoint_path = None
    if args.in_checkpoint:
        checkpoint_path = args.in_checkpoints
    elif args.start_iteration:
        checkpoint_path = args.checkpoint_dir / f"checkpoint_{args.start_iteration:06d}.pth"
    
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))

    model.train()
    model.to(args.device)

    optimizer = optimizer_factory(model, args.opt_config)

    logging.info(f"Model: {model}")
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
