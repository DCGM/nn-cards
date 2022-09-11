# file train_layout.py
# author Michal Hradiš, Kristína Hostačná

import os
import json
import numpy as np
import pandas as pd
import argparse
import logging

import cv2
import re
import networkx as nx

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from shapely.geometry import LineString, Point
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, KBinsDiscretizer, FunctionTransformer

from nnets import net_factory


def parse_arguments(arguments):
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
    parser.add_argument('--img-path',  type=str, help="Save images with graphs.")
    parser.add_argument('--aug',  type=str, help="Augumentation config")

    args = parser.parse_args(arguments)
    return args

def parse_affine(scale=None, translate_percent=None, translate_px=None, rotate=None, shear=None):
    affine_res=iaa.Affine(scale=scale, translate_percent=translate_percent, translate_px=translate_px, rotate=rotate, shear=shear)
    return affine_res
    
def parse_augumentation(Affine=None, Multiply=None, Fliplr=None, Flipud=None, 
                        GaussianBlur=None, Crop=None, AddToHueAndSaturation=None, 
                        AdditiveGaussianNoise=None, Sharpen=None, SigmoidContrast=None):
    if Affine is not None :
        Affine=parse_affine(**Affine)
    if Multiply is not None :
        Multiply=iaa.Multiply(Multiply)
    if Fliplr is not None :
        Fliplr=iaa.Fliplr(Fliplr)
    if Flipud is not None :
        Flipud=iaa.Flipud(Flipud)
    if GaussianBlur is not None :
        GaussianBlur=iaa.GaussianBlur(GaussianBlur)
    if Crop is not None :
        Crop=iaa.Crop(Crop)
    if AddToHueAndSaturation is not None :
        AddToHueAndSaturation=iaa.AddToHueAndSaturation(AddToHueAndSaturation)
    if AdditiveGaussianNoise is not None :
        AdditiveGaussianNoise=iaa.AdditiveGaussianNoise(AdditiveGaussianNoise)
    if Sharpen is not None :
        Sharpen=iaa.Sharpen(Sharpen)
    if SigmoidContrast is not None :
        SigmoidContrast=iaa.SigmoidContrast(SigmoidContrast)
    parameters=[Affine, Multiply, Fliplr, Flipud, GaussianBlur, Crop, AddToHueAndSaturation,AdditiveGaussianNoise, Sharpen, SigmoidContrast]
    augumenters=[aug for aug in parameters if aug is not None]
    seq = iaa.Sequential(augumenters)
    return seq

def augument(inputs, seq):
#     TODO: add width and height of img (change hardcoded values)
    for idx, points in enumerate(inputs):
        start=Keypoint(x=points[0], y=points[1])
        end=Keypoint(x=points[2], y=points[3])
        image = ia.quokka(size=(1744, 1240))

        image_aug, points_aug = seq(image=image, keypoints=[start,end])

        # cut off endpoints of lines at frame
        picture_frame = LineString([(0,0),(0,1744),(1240,1744),(1240,0),(0,0)])
        line = LineString([(points_aug[0].x, points_aug[0].y), (points_aug[1].x, points_aug[1].y)])

        intersects= line.intersection(picture_frame)
        # replace both startpoint and endpoint
        if isinstance(intersects, list):
            points_aug[0].x = intersects[0].x
            points_aug[0].y = intersects[0].y
            points_aug[1].x = intersects[1].x
            points_aug[1].y = intersects[1].y
        elif isinstance(intersects, Point):
            # replace startpoint
            if points_aug[0].x <= 0 or points_aug[0].x >= 1240 or points_aug[0].y <= 0 or points_aug[0].y >= 1744:
                points_aug[0].x = intersects.x
                points_aug[0].y = intersects.y

            # replace endpoint
            else:
                points_aug[1].x = intersects.x
                points_aug[1].y = intersects.y
        inputs[idx]=[points_aug[0].x, points_aug[0].y, points_aug[1].x, points_aug[1].y]
    return inputs

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

def preprocess_data(data_frame, aug=None):
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
    
    if aug is not None:
        inputs = augument(inputs, aug)
        
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

def get_dataframe(data_path):
    data = pd.read_csv(data_path, converters={'startX': float, 'startY': float,
                                              'endX': float, 'endY': float})
    df = pd.DataFrame(data)
    return df

def load_data(data_path, k_nearest, aug=None):
    df= get_dataframe(data_path)
    inputs,oh_labels = preprocess_data(df,aug)
    
    card_starts_indices, _ = split_to_cards(df)
#     n_of_cards_training = 50 # check if necessary
    graphs = data_to_graphs(card_starts_indices, inputs, oh_labels, k_nearest)
    
    return graphs

def graph_to_image(path, graph, thickness=2, circles=True):
    img = cv2.imread(path)
    print(graph)
    center_list=nx.get_node_attributes(graph, "pos")
    lines=[]
    color=(0, 0, 0)
    
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

def get_graph_imgs(csv_path):
    graphs = load_data(csv_path, 8)#csv_path

    df= get_dataframe(csv_path)    
    _,card_names = split_to_cards(df)
    return (card_names,graphs)

def save_graph_imgs(imgs_path, csv_path):
    card_names,graphs= get_graph_imgs(csv_path)
    
    for card_idx, card in enumerate(card_names):
        graph = to_networkx(graphs[card_idx], node_attrs=["x", "pos"], to_undirected=True)
        
        card_name=re.sub("[^A-Za-z-_.]","",card_names[card_idx][0])
        card_img_path=imgs_path+card_name
        img= image.imread(card_img_path)
        # img
        if not os.path.exists(card_img_path + "/withGraph"):
            os.makedirs(card_img_path + "/withGraph")
            
        new_file_name = card_img_path + "withGraph/" + card.name
        img = graph_to_image(card_img_path, graph, circles=True)
        
        cv2.imwrite(new_file_name, img)

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

    split_idx = 50
    
    graphs = load_data(args.data_path, k_nearest=args.k_nearest)
    
    if args.img_path is not None:
        save_graph_imgs(args.img_path, args.data_path)
    if args.aug is not None:
        seq_config = json.loads(args.aug)
        seq= parse_augumentation(**seq_config)
        graphs_aug = load_data(args.data_path, k_nearest=args.k_nearest, aug=seq)
        
        # TODO: def load_data_aug (to ignore train data during processing)
        train_dataset = graphs[split_idx:] + graphs_aug[split_idx:]
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




