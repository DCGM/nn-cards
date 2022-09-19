# file train_layout.py
# author Michal Hradiš, Kristína Hostačná

import os
import json
import numpy as np
import pandas as pd
import argparse
import logging

import matplotlib.pyplot as plt
import cv2
import networkx as nx

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from shapely.geometry import LineString, Point
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, KBinsDiscretizer, FunctionTransformer

from nets import net_factory
from lineToLabel import CardMeta
from augumentations import augument_img, normalize_inputs, keypoints_to_array, parse_augumentation, augument

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
    parser.add_argument('--imgs-list',  type=str, help="List of paths to images to be saved after augumetation.")
    parser.add_argument('--imgs-dst',  type=str, default="./",  help="Path to save augumented images to.") # "/withGraph"
    parser.add_argument('--aug-num',  type=int, default=1, help="Number of augumented images to generate.")
    parser.add_argument('--aug',  type=str, help="Augumentation config.")

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

def split_to_cards(data_frame):
    cards = data_frame.cardName.to_numpy().reshape(-1, 1)
    card_indices = np.unique(cards, return_index=True)

    card_names=[(x,y) for (x, y) in sorted(zip(card_indices[0], card_indices[1]), key=lambda pair: pair[1])]

    card_indices[1].sort()

    card_starts_indices = card_indices[1]
    card_starts_indices = np.append(card_starts_indices, len(cards))

    return (card_starts_indices, card_names)

def preprocess_data(data_frame):
    startX = data_frame.startX.to_numpy().reshape(-1, 1)
    startY = data_frame.startY.to_numpy().reshape(-1, 1)
    endX = data_frame.endX.to_numpy().reshape(-1, 1)
    endY = data_frame.endY.to_numpy().reshape(-1, 1)

    labels_data = np.array(data_frame.label).reshape(-1, 1)

    int_encoder = LabelEncoder()
    labels = int_encoder.fit_transform(labels_data.ravel()).reshape(-1, 1)

    onehot_encoder = OneHotEncoder(sparse=False).fit(labels)
    oh_labels = onehot_encoder.transform(labels)

    inputs = np.concatenate((startX, startY, endX, endY), axis=1)

    # normalize inputs
    inputs[..., 0] /= 1200
    inputs[..., 1] /= 1700
    inputs[..., 2] /= 1200
    inputs[..., 3] /= 1700

    return (inputs,oh_labels)

def data_to_graphs(card_starts_indices, inputs, oh_labels, k_nearest):
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

def create_modified_graph(graph, inputs=None, outputs=None, edges=None):
    if inputs is None:
        inputs=graph['x']
    if outputs is None:
        outputs=graph['y']
    if edges is None:
        edges=graph['edge_index']
    pos = []
    for idx, center in enumerate(inputs):
        pos.append(((inputs[idx][0] + inputs[idx][2]) / 2, (inputs[idx][1] + inputs[idx][3]) / 2))
    pos = torch.tensor(np.array(pos), dtype=torch.float)
    modified_graph= Data(x=inputs, y=outputs, pos=pos, edge_index=edges)
    return modified_graph

def get_dataframe(data_path):
    data = pd.read_csv(data_path, converters={'startX': float, 'startY': float,
                                              'endX': float, 'endY': float,
                                              'cardHeight': int, 'cardWidth': int})
    df = pd.DataFrame(data)
    return df

def load_data(data_path, k_nearest, aug=None, aug_num=1, n_of_cards_training = 0):
    # TODO: split by cards, THEN augument, THEN create graphs
    df= get_dataframe(data_path)

    card_starts_indices, _ = split_to_cards(df)

    inputs,oh_labels = preprocess_data(df)

    graphs = data_to_graphs(card_starts_indices, inputs, oh_labels, k_nearest)

    return graphs

def graph_to_image(img, graph, thickness=2, circles=True):
    center_list=nx.get_node_attributes(graph, "pos")

    color=(255, 0, 0)

    for u,v in graph.edges:
        x1= int(np.round(center_list[u][0]*1200))
        y1= int(np.round(center_list[u][1]*1700))

        x2= int(np.round(center_list[v][0]*1200))
        y2= int(np.round(center_list[v][1]*1700))

        cv2.line(img, (x1,y1),(x2,y2), color, thickness)
        if circles:
            cv2.circle(img, (x1,y1), 3, color, 4)
            cv2.circle(img, (x2,y2), 3, color, 4)
    return img

def get_og_graph_imgs(csv_path):
    graphs = load_data(csv_path, 8, aug=None, aug_num=1, n_of_cards_training = 0)#csv_path

    df= get_dataframe(csv_path)
    _,card_names = split_to_cards(df)
    return (card_names,graphs)

def save_graph_img(img, graph, dst_path):
    img = graph_to_image(img, graph, circles=True)
    cv2.imwrite(dst_path, img)

def find_img (card_path):
    if not os.path.exists(os.path.join(card_path)):
        return None
    else:
        img = cv2.imread(card_path)
        return img

def save_graph_imgs(src_path, card_names, graphs, dst_path):
    for img_path in src_path:
        img=find_img(img_path)
        if img is None:
            print("IMG "+img_path+ " not found")
            continue

    for card_idx, card in enumerate(card_names):
        graph = to_networkx(graphs[card_idx], node_attrs=["x", "pos"], to_undirected=True)

        card_name=os.path.basename(card_names[card_idx][0])

        # img
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        new_file_name = os.path.join(dst_path, card.name)
        save_graph_img(img, graph, new_file_name)

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

def load_cards(csv_path, k_neighbours=8):
    graphs = load_data(csv_path, k_neighbours, aug=None, aug_num=1, n_of_cards_training = 0)

    df= get_dataframe(csv_path)
    heights = df.cardHeight.to_numpy().reshape(-1, 1)
    widths = df.cardWidth.to_numpy().reshape(-1, 1)

    card_starts_indices,card_names = split_to_cards(df)
    cards=[]
    for idx,card_name in enumerate(card_names):
        df_idx=card_starts_indices[idx]

        card = CardMeta(card_name[0], int(heights[df_idx]), int(widths[df_idx]))

        g = to_networkx(graphs[idx], node_attrs=["x" ,"y", "pos"], to_undirected=True)
        card.add_graphs(graphs[idx],g)

        cards.append(card)
    return cards

def main():
    print("START")
    args = parse_arguments()

    split_idx = 50

    cards= load_cards(args.data_path, 8)
    graphs = load_data(args.data_path, k_nearest=args.k_nearest)
    if args.aug is not None:
        seq_config = json.loads(args.aug)
        seq= parse_augumentation(**seq_config)


        if args.imgs_list is not None:
            # get img_names
            img_names={}
            if type(args.imgs_list) == list:
                for card_path in args.imgs_list:
                    img_names[os.path.basename(card_path)] = card_path
            else:
                img_names[os.path.basename(args.imgs_list)] = args.imgs_list

            augumented_graphs = []
            for idx, card in enumerate(cards):
                if card.name in img_names:
                    img = cv2.imread(img_names[card.name])
                    card.add_img(img)

                augumented_data = augument_img(card, seq, args.aug_num)  # list[(augumented_img, augumented_coords),...]

                counter=0
                for (augumented_img, augumented_coords) in augumented_data:
                    if idx >= split_idx:
                        inputs=keypoints_to_array(augumented_coords)
                        inputs= normalize_inputs(inputs,(card.height,card.width))
                        inputs=torch.tensor(np.array(inputs), dtype=torch.float)
                        aug_graph= create_modified_graph(card.graph, inputs=inputs)
                        augumented_graphs.append(aug_graph)
                        if card.img is not None:
                            if counter==0:
                                img = graph_to_image(card.img, card.nx_graph)
                                cv2.imwrite(os.path.join(args.imgs_dst, card.name), img)
                            aug_g = to_networkx(aug_graph, node_attrs=["x" , "pos"], to_undirected=True)

                            img = graph_to_image(augumented_img, aug_g)
                            dst_path=os.path.join(args.imgs_dst, str(counter)+ card.name)

                            cv2.imwrite(dst_path, img)
                            counter += 1

        train_dataset = graphs[split_idx:] + augumented_graphs
    else:
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