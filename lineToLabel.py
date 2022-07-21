#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import numpy as np
import os
import argparse
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import json
import csv

from shapely.geometry import Point
from shapely.geometry import Polygon
from matplotlib.pyplot import figure
from pero_ocr.document_ocr.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser
# TODO: parseargs


class CardMeta:
    def __init__(self, name, card_id, height, width):
        self.name = name
        self.id = card_id
        self.height = height
        self.width = width
        # TODO add img??
        # TODO add lines (label,line) tuple
        # self.polygons = []  # TODO: def add_polygon (label,polygon) tuple
# TODO: private parameters+ functions get


def save_result(result):
    out_name = "lineToLabel"
    # TODO overwrite
    # TODO make json?
    with open(out_name+'.csv', 'w') as f:
        writer = csv.writer(f)
        for row in result:
            writer.writerow(row)


def map_text_area_to_labels(text_area, card, polygons):
    # TODO  textarea = line. self.polygon = polygon/logit_coords?
    pass


def most_numerous_polygon(polygon_list):
    point_count = {}
    for i in set(polygon_list):
        x=polygon_list.count(i)
        if i in point_count:
            point_count[x].append(i)
        else:
            point_count[x] = [i]
    # ! if same number of points belongs to 2+ polygons, only saving 1 of them
    return point_count[max(point_count)]


def map_lines_to_labels(lines, card, polygons):
    result = []
    for line in lines:
        number_of_points = len(line.baseline)
        in_polygons = []
        for point in line.baseline:
            # TODO optimization? (from shapely.ops import nearest_points)
            # TODO optimization save found polygon
            shapely_point = Point(point[0], point[1])
            for label, polygon in polygons:
                if polygon.contains(shapely_point):
                    in_polygons.append(label[0])  # TODO check why is label list
        if len(in_polygons) >= number_of_points/2:  # if more than half points is in polygon
            result_label = most_numerous_polygon(in_polygons)
            result.append((line.id, result_label[0]))  # TODO check why is label list
        else:
            result.append((line.id, "None"))
    return result


def render_polygons(img, card, polygons):
    fig, axs = plt.subplots(sharex=True, sharey=True)
    axs.set_ylim(axs.get_ylim()[::-1])
    axs.axis('equal')
    for label, polygon in polygons:
        x, y = polygon.exterior.xy
        axs.plot(x, y, label=label)
    axs.imshow(img, aspect='auto')

    fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.7)

    plt.show()


def render_xml(img, layout):
    render = layout.render_to_image(img)

    fig = plt.figure()
    fig.tight_layout()
    fig.subplots_adjust(right=0.7)

    plt.imshow(render, aspect='auto')
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-j', '--json', required=True, help='Path to json file .')
    args = parser.parse_args()

    return args


def parse_json(json_file_path):
    json_annot = ""

    # TODO: read whole json
    with open(json_file_path) as f:
        label_studio_json = json.load(f)
        for element in label_studio_json:
            if element['id'] == 1655:
                json_annot = element

    # TODO: read size(axes)
    polygons = []
    for annotation_polygon in json_annot["annotations"][0]["result"]:
        coords = []
        height = json_annot["annotations"][0]["result"][0]["original_height"]
        width = json_annot["annotations"][0]["result"][0]["original_width"]
        card = CardMeta(json_annot["file_upload"], json_annot["id"], height, width)

        label = annotation_polygon["value"]["polygonlabels"]
        for point in annotation_polygon["value"]["points"]:
            x = point[0] * width / 100
            y = point[1] * height / 100
            coords.append((x, y))

        polygon = Polygon(coords)
        polygons.append((label, polygon))
    return card, polygons


def parse_csv(csv_file_path):
    label_studio_csv = pd.read_csv('./data/proto/00008.csv')
    print(label_studio_csv)


def main():
    # initialize some parameters
    args = parse_arguments()
    results = []
    # TODO: process multiple cards
    # load jpg
    img = cv2.imread('./data/proto/00008.jpg')

    # load xml
    layout = PageLayout(5, page_size=(1240, 1744), file="./data/proto/00008.xml")
    lines = list(layout.lines_iterator())

    # # load json
    card, polygons = parse_json("./data/proto/all.json")

    # # load csv
    # parse_csv('./data/proto/00008.csv')

    results = map_lines_to_labels(lines, card, polygons)
    print(results)
    save_result(results)

    render_polygons(img, card, polygons)
    render_xml(img, layout)


if __name__ == "__main__":
    main()

