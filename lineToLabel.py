#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import cv2
import matplotlib.pyplot as plt
import json
import csv

from shapely.geometry import Point
from shapely.geometry import Polygon
from matplotlib.pyplot import figure
from pero_ocr.document_ocr.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser
# -i ./data/annot/jpg/ -x ./data/annot/xml/ -j ./data/annot/all.json -o scriptOutput


class CardMeta:
    def __init__(self, name, card_id, height, width):
        self.name = name
        self.id = card_id
        self.height = height
        self.width = width
        # TODO add img??
        # TODO add lines (label,line) tuple
        # self.polygons = []  # TODO: def add_polygon (label,polygon) tuple


def save_result(results, out_name):
    # TODO overwrite
    with open(out_name+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(("lineID", "label", "cardName", "points", "lineNumber"))
        for row in results:
            writer.writerow(row)


def extract_filename_label_studio(file_upload):
    filename_arr = file_upload.split("-")
    filename = '-'.join(filename_arr[1:])
    return filename


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
    # ^ feature. not a bug
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
            result.append((line.id, result_label[0], card.name))  # TODO check why is label list
        else:
            result.append((line.id, "None", card.name))
    return result


def extract_features(lines, card):
    features = []
    for index, line in enumerate(lines):
        points = [tuple(line.baseline[0]), tuple(line.baseline[-1])]

        features.append((points, index+1))
    return features


def render_polygons(img, card, polygons):
    # todo rewrite for cv
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


def match_json_to_xml(json_file_path, xml_file_path):
    files_in_json = []
    xml_files = []
    # read json
    with open(json_file_path) as f:
        label_studio_json = json.load(f)
        for json_count, element in enumerate(label_studio_json):
            # extract filenames
            filename = extract_filename_label_studio(element["file_upload"])
            files_in_json.append(filename[:-4])  # delete extension
        json_count += 1  # index to count

    # read all xml
    xml_count = 0
    if os.path.isfile(xml_file_path):
        xml_files.append(xml_file_path)
    if os.path.isdir(xml_file_path):
        for file in os.listdir(xml_file_path):
            if file.endswith(".xml"):
                xml_count += 1
                xml_files.append(file[:-4])  # delete extension

    matching = [x for x in xml_files if x in files_in_json]

    return matching, json_count-len(matching), xml_count-len(matching)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', required=True, help='Path to json file '
                                                            '(containing label studio polygons).')
    parser.add_argument('-x', '--xml', required=True, help='Path to directory with xml files '
                                                           '(containing page layout).')
    parser.add_argument('-i', '--img', required=False, help='Path to directory with img files.')
    parser.add_argument('-o', '--output', required=False, default="lineToLabelOutput", help='Name of output file.' )

    args = parser.parse_args()

    # todo check extensions
    if not os.path.exists(args.xml):
        print("Couldn't find XML you entered")
        sys.exit(args.xml)  # TODO check clean code solution
    if not os.path.exists(args.json):
        print("Couldn't find JSON you entered")
        sys.exit(args.json)  # TODO check clean code solution

    return args


def parse_json(json_file_path):
    json_annot = ""

    # TODO: read whole json
    with open(json_file_path) as f:
        label_studio_json = json.load(f)
        for element in label_studio_json:
            if element['id'] == 1655:
                json_annot = element

    polygons = []
    for annotation_polygon in json_annot["annotations"][0]["result"]:
        coords = []
        height = json_annot["annotations"][0]["result"][0]["original_height"]
        width = json_annot["annotations"][0]["result"][0]["original_width"]
        name = extract_filename_label_studio(json_annot["file_upload"])
        card = CardMeta(name, json_annot["id"], height, width)

        label = annotation_polygon["value"]["polygonlabels"]
        for point in annotation_polygon["value"]["points"]:
            x = point[0] * width / 100
            y = point[1] * height / 100
            coords.append((x, y))

        polygon = Polygon(coords)
        polygons.append((label, polygon))
    return card, polygons


def main():
    results = []
    layouts = []

    # initialize parameters
    args = parse_arguments()

    # load jpg
    img = cv2.imread(args.img)  # TODO: load all img paths

    # load xmls
    matched_files, unmatched_json, unmatched_xml = match_json_to_xml(args.json, args.xml)

    if unmatched_json > 0:
        print("Warning, "+str(unmatched_json)+" files in JSON don't have corresponding XML files.")
    if unmatched_xml > 0:
        print("Warning, "+str(unmatched_xml)+" XML files don't have corresponding files in JSON.")

    for xml in matched_files:
        filepath = args.xml + xml + ".xml"  # todo check (/+extract) xml-> dir
        layouts.append(PageLayout(5, page_size=(1240, 1744), file=filepath))

    # load json
    card, polygons = parse_json(args.json)  # todo string -> args

    for layout in layouts:
        lines = list(layout.lines_iterator())
        gt = map_lines_to_labels(lines, card, polygons)  # lineID, label, cardName
        features = extract_features(lines, card)  # [(x1,y1),(xN,yN)], LineNumber
        for index, ground_truth in enumerate(gt):
            results.append(ground_truth + features[index])
    save_result(results, args.output)

    # render_polygons(img, card, polygons)
    # render_xml(img, layout)


if __name__ == "__main__":
    main()

