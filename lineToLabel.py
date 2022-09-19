#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import cv2
import json
import csv
import numpy as np

from shapely.geometry import Polygon, Point, mapping
from pero_ocr.document_ocr.layout import PageLayout, draw_lines
from pero_ocr.document_ocr.page_parser import PageParser


# -i ./data/annot/jpg/ -x ./data/annot/xml/ -j ./data/annot/all.json -o scriptOutput


class CardMeta:
    def __init__(self, name,  height, width, card_id="", polygons=None):
        self.img = None
        self.path = None
        self.layout = None
        self.graph = None
        self.labeled_lines = []  # list of (baseline, id, label)

        self.name = name
        self.id = card_id
        self.height = height
        self.width = width
        self.polygons = polygons
        # todo if not none (render_to_image)

    def add_line(self, lines_and_labels):
        self.labeled_lines = self.labeled_lines + lines_and_labels

    def add_layout(self, layout):
        self.layout = layout
    def add_graph(self, graph):
        self.graph = graph

    def add_path(self, path):
        self.path = path

    def render_to_image(self, image, thickness=2, circles=True):
        """Render labeled lines and polygons to image.
        :param image: image to render layout into
        """
        categories = {"Name": (0, 143, 122),
                      "None": (0, 129, 207),
                      "Rank": (132, 94, 194),
                      "Birth": (214, 93, 177),
                      "Nationality": (255, 0, 0),
                      "Died": (255, 111, 145),
                      "Buried-date": (255, 150, 113),
                      "Buried-place": (255, 199, 95),
                      "Grave-position": (249, 248, 113),
                      "Source-place": (0, 201, 167),
                      "Source-book": (31, 46, 126)}

        for line in self.layout.lines_iterator():
            line_gt = list(filter(lambda x: x[1] == line.id, self.labeled_lines))[0]
            label = line_gt[2]
            color = categories[label] if label in categories else (0, 0, 0)

            # line
            image = draw_lines(
                image,
                [line.baseline], color=color,
                circles=(circles, circles, False), thickness=thickness)
            # label
            cv2.putText(image, label, line.baseline[-1], cv2.FONT_HERSHEY_PLAIN, 2,
                        color=color, thickness=2, lineType=cv2.LINE_AA)

        for label, polygon in self.polygons:
            coords = mapping(polygon)['coordinates']
            if coords is not None:
                image = draw_lines(
                    image,
                    coords, color=(255, 0, 0), close=True,
                    thickness=thickness)

        return image


def save_result(results, out_name):
    with open(out_name + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(("lineID", "label", "cardName", "startX", "startY", "endX", "endY", "cardHeight", "cardWidth"))
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
        x = polygon_list.count(i)
        if i in point_count:
            point_count[x].append(i)
        else:
            point_count[x] = [i]
    # ! if same number of points belongs to 2+ polygons, only saving 1 of them
    # ^ feature. not a bug
    return point_count[max(point_count)]


# todo rewrite as function of card
def map_lines_to_labels(layout, card):
    lines = list(layout.lines_iterator())  #
    result = []
    for line in lines:
        number_of_points = len(line.baseline)
        in_polygons = []
        for point in line.baseline:
            shapely_point = Point(point[0], point[1])
            for label, polygon in card.polygons:
                if polygon.contains(shapely_point) and label[0] != "Printed":
                    in_polygons.append(label[0])  # TODO check why is label list
        if len(in_polygons) >= number_of_points / 2:  # if more than half points is in polygon
            result_label = most_numerous_polygon(in_polygons)
            result.append((line.baseline, line.id, result_label[0]))  # TODO check why is label list
        else:
            result.append((line.baseline, line.id, "None"))

    if layout.id == card.name:
        card.add_line(result)
    else:
        print("error mapping lines to labels - names not matching:"+layout.id, card.name)

    return result


def extract_features(lines, card):
    features = []
    for index, line in enumerate(lines):
        start_x = line.baseline[0][0]
        start_y = line.baseline[0][1]
        end_x = line.baseline[-1][0]
        end_y = line.baseline[-1][1]

        # start_point = str(line.baseline[0][0])+","+str(line.baseline[0][1])
        # end_point = str(line.baseline[-1][0])+","+str(line.baseline[-1][1])
        features.append((start_x, start_y, end_x, end_y, index + 1))
    return features


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
    xml_extra_count = 0
    if os.path.isfile(xml_file_path):
        xml_files.append(xml_file_path)
    if os.path.isdir(xml_file_path):
        for file in os.listdir(xml_file_path):
            if file.endswith(".xml"):
                if file[:-4] in files_in_json:  # only caring about matching files
                    xml_files.append(file[:-4])  # delete extension
                else:
                    xml_extra_count += 1

    matching = [x for x in files_in_json if x in xml_files]
    json_extra = set(files_in_json) - set(xml_files)

    return matching, json_count - len(matching), xml_extra_count


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', required=True, help='Path to json file '
                                                            '(containing label studio polygons).')
    parser.add_argument('-x', '--xml', required=True, help='Path to directory with xml files '
                                                           '(containing page layout).')
    parser.add_argument('-i', '--img', required=False, help='Path to directory with img files.')
    parser.add_argument('-o', '--output', required=False, default="lineToLabelOutput", help='Name of output file.')

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
    json_annot = []
    with open(json_file_path) as f:
        label_studio_json = json.load(f)
        for card in label_studio_json:
            json_annot.append(card)

    cards = []
    for card in json_annot:
        polygons = []

        height = card["annotations"][0]["result"][0]["original_height"]
        width = card["annotations"][0]["result"][0]["original_width"]
        name = extract_filename_label_studio(card["file_upload"])

        for annotation_polygon in card["annotations"][0]["result"]:
            coords = []

            label = annotation_polygon["value"]["polygonlabels"]
            for point in annotation_polygon["value"]["points"]:
                x = point[0] * width / 100
                y = point[1] * height / 100
                coords.append([x, y])

            polygon = Polygon(coords)
            polygons.append((label, polygon))

        card = CardMeta(name, height, width, card_id=card["id"], polygons=polygons)
        cards.append(card)
    return cards


# todo
def load_files(path):
    # check if exist
    # check if file/dir
    # return list
    pass


def visualize(path, cards, idx):
    img = cv2.imread(path)
    im = cards[idx].render_to_image(img)

    # plt.imshow(im, aspect='auto')
    # plt.show()
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_data_to_cards(json_path,layouts, img_path):
    # load json
    cards = parse_json(json_path)

    for idx, layout in enumerate(layouts):    # todo make sure that layout is mapped to card
        map_lines_to_labels(layout, cards[idx])  # line.baseline, lineID, label >card
        cards[idx].add_layout(layout)
        cards[idx].add_path(img_path + cards[idx].name)
    return cards

def get_layouts(matched_files,xml_path):
    layouts = []
    for xml in matched_files:
        filepath = os.path.join(xml_path + xml + ".xml") # todo check (/+extract) xml-> dir
        layouts.append(PageLayout(5, page_size=(1240, 1744), file=filepath))
    return  layouts


def main():
    results = []

    # initialize parameters
    args = parse_arguments()

    # load xmls
    matched_files, unmatched_json, unmatched_xml = match_json_to_xml(args.json, args.xml)

    if unmatched_json > 0:
        print("Warning, " + str(unmatched_json) + " files in JSON don't have corresponding XML files.")
    if unmatched_xml > 0:
        print("Warning, " + str(unmatched_xml) + " XML files don't have corresponding files in JSON.")

    layouts=get_layouts(matched_files,args.xml)
    cards=load_data_to_cards(args.json,layouts,args.img)

    # save results
    for card in cards:
        # csv
        size=[card.height, card.width]
        for line in card.labeled_lines:
            baseline = line[0]
            coords = [baseline[0][0], baseline[0][1], baseline[-1][0], baseline[-1][1]]  # x1, y1, xN, yN
            features = [line[1], line[2], card.name]  # lineID, label, cardName,
            results.append((features + coords + size))
        # img
        if not os.path.exists(args.img + "/labeled"):
            os.makedirs(args.img + "/labeled")
        filename = args.img + "labeled/" + card.name
        img = card.render_to_image(cv2.imread(card.path))
        cv2.imwrite(filename, img)

    save_result(results, args.output)

    # visualization
    idx = 50
    # visualize(cards[idx].path, cards, idx)
    print("end")
    # todo comments

if __name__ == "__main__":
    main()
