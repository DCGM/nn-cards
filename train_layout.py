# file train_layout.py
# author Michal Hradiš, Kristína Hostačná

import os
import json
import numpy as np
import pandas as pd
import argparse
import logging

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from shapely.geometry import LineString, Point
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, KBinsDiscretizer, FunctionTransformer

from nets import net_factory


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name', help='Model name', required=True)
    parser.add_argument(
        '-d', '--data-path', help='Path to data csv file', required=True)
    parser.add_argument('--start-iteration', default=0, type=int)
    parser.add_argument('--max-iterations', default=50000, type=int)
    parser.add_argument('--view-step', default=1000, type=int)
    parser.add_argument('-i', '--in-checkpoint', type=str)
    parser.add_argument('-o', '--out-checkpoint', type=str)
    parser.add_argument('--checkpoint-dir', default='.', type=str)
    parser.add_argument('-g', '--gpu-id', type=int,
                        help="If not set setGPU() is called. Set this to 0 on desktop. Leave it empty on SGE.")
    parser.add_argument('--batch-size', default=32, type=int, help="Batch size.")
    parser.add_argument('--learning-rate', type=float, default=0.0002, help="Learning rate for ADAM.")
    parser.add_argument('--net-config', default='{"type": "mlp", "hidden_dim": 128, "depth": 4}',
                        help="Json network config.")
    parser.add_argument('--optimization-config', default='{"type":"Adam"}')
    parser.add_argument('--k-nearest', default=4, type=int, help="K nearest neighbors when building the graph.")

    args = parser.parse_args()
    return args


def find_neighbours(line, card_lines, k_nearest=4):
    shapely_line = LineString([Point(line[0], line[1]), Point(line[2], line[3])])
    lines_distances = []
    for neighbour in card_lines:
        line_center = Point((neighbour[0] + neighbour[2]) / 2, (neighbour[1] + neighbour[3]) / 2)
        lines_distances.append(line_center.distance(shapely_line))

    res = {distance: index for index, distance in enumerate(lines_distances)}
    res = dict(sorted(res.items()))

    return list(res.items())[:k_nearest]


def load_data(data_path, k_nearest):
    data = pd.read_csv(data_path, converters={'startX': float, 'startY': float,
                                              'endX': float, 'endY': float})
    df = pd.DataFrame(data)

    # Preprocess data
    startX = df.startX.to_numpy().reshape(-1, 1) / 1200
    startY = df.startY.to_numpy().reshape(-1, 1) / 1700
    endX = df.endX.to_numpy().reshape(-1, 1) / 1200
    endY = df.endY.to_numpy().reshape(-1, 1) / 1700

    labels_data = np.array(df.label).reshape(-1, 1)

    int_encoder = LabelEncoder()
    labels = int_encoder.fit_transform(labels_data.ravel()).reshape(-1, 1)

    onehot_encoder = OneHotEncoder(sparse=False).fit(labels)
    oh_labels = onehot_encoder.transform(labels)

    inputs = np.concatenate((startX, startY, endX, endY), axis=1)

    n_of_cards_training = 50
    cards = df.cardName.to_numpy().reshape(-1, 1)
    card_indices = np.unique(cards, return_index=True)
    card_indices[1].sort()

    card_starts_indices = card_indices[1]

    card_starts_indices = np.append(card_starts_indices, len(cards))

    graphs = []
    # for each card
    for index, card in enumerate(card_starts_indices):
        #  split by cards
        if index < len(card_starts_indices) - 1:
            lines = []
            edges = []
            target = []
            center = []
            for line in range(card, card_starts_indices[index + 1]):
                center.append(((inputs[line][0] + inputs[line][2]) / 2, (inputs[line][1] + inputs[line][3]) / 2))
                lines.append(inputs[line])
                target.append(oh_labels[line])
                line_neigh = find_neighbours(inputs[line], inputs[card: card_starts_indices[
                    index + 1]], k_nearest=k_nearest)
                for neigh_distance, neigh_index in line_neigh:
                    edges.append([line - card, neigh_index])
            nodes_lines = torch.tensor(np.array(lines), dtype=torch.float)
            edge_index = torch.tensor(np.array(edges), dtype=torch.long)
            pos = torch.tensor(np.array(center), dtype=torch.float)
            targets = torch.tensor(np.array(target), dtype=torch.float)

            graphs.append(Data(x=nodes_lines, y=targets, pos=pos, edge_index=edge_index.t().contiguous()))

    return graphs


def test(data_loader, model):
    model = model.eval()

    # Accumulators
    loss_acc = 0
    correct = 0
    counter = 0
    batch_items = 0

    # Loop through dataset
    for batch_data in data_loader:
        batch_data = batch_data.to(next(model.parameters()).device)

        # Calculate output
        output = model(batch_data)

        # Accumulate loss value
        loss_acc += F.cross_entropy(output, batch_data.y).item()

        # Calculate accuracy
        pred_y = torch.argmax(output,dim=1)
        label = torch.argmax(batch_data.y,dim=1)
        correct += (pred_y == label).sum().item()

        # Accumulate batch size (graphs don't have constant number of nodes)
        counter += batch_data.x.shape[0]
        batch_items += 1

    model.train()

    return loss_acc / batch_items, correct / counter


def main():
    print("START")
    args = parse_arguments()

    graphs = load_data(args.data_path, k_nearest=args.k_nearest)

    split_idx = 50
    train_dataset = graphs[split_idx:]
    test_dataset = graphs[:split_idx]

    # Create training and testing DataLoaders
    training_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    testing_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    config = json.loads(args.net_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net_factory(config, input_dim=4, output_dim=11)

    checkpoint_path = None
    if args.in_checkpoint is not None:
        checkpoint_path = args.in_checkpoint
    elif args.start_iteration:
        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_{:06d}.pth".format(args.start_iteration))
    if checkpoint_path is not None:
        logging.info(f"Restore {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)

    optim_config = json.loads(args.optimization_config)
    optim_type = optim_config['type'].lower()
    del optim_config['type']

    if optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, **optim_config)
    elif optim_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, **optim_config)


    logging.info('Start')
    loss_list = []
    iteration = args.start_iteration
    while iteration < args.max_iterations:
        for data in training_loader:
            iteration += 1
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, data.y).mean()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            if iteration % args.view_step == 0:
                if args.out_checkpoint is not None:
                    checkpoint_path = args.out_checkpoint
                else:
                    if not os.path.exists(args.checkpoint_dir):
                        os.makedirs(args.checkpoint_dir)
                    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_{:06d}.pth".format(iteration + 1))
                torch.save(model.state_dict(), checkpoint_path)

                # Calculate loss and accuracy on the testing dataset
                test_loss_acc, acc = test(testing_loader, model)

                print(f"accuracy_score:{acc:.3f} "
                    f"iteration:{iteration} "
                    f"train_loss:{np.mean(loss_list):.3f} "
                    f"test_loss:{test_loss_acc:.3f} ")

                loss_list = []

            # Stop training when amount of iterations is reached
            if iteration >= args.max_iterations:
                break


if __name__ == "__main__":
    main()




