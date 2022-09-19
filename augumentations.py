# file augumentations.py
# author Kristína Hostačná

import logging

# import imgaug as ia
import networkx as nx
import numpy as np
import imgaug.augmenters as iaa

from imgaug.augmentables import Keypoint, KeypointsOnImage
# from lineToLabel import CardMeta

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

def augument_img(card, aug, aug_num=1):
    if card.img is not None:
        img = card.img
    else:
        img_shape= (card.height,card.width)
        img=np.zeros(img_shape, np.uint8)
    graph_x=nx.get_node_attributes(card.graph, "x").values()

    kps_list= []
    for idx, points in enumerate(graph_x):

        start=Keypoint(x=points[0]*card.width, y=points[1]*card.height)
        end=Keypoint(x=points[2]*card.width, y=points[3]*card.height)
        kps_list.append(start)
        kps_list.append(end)

    kps= KeypointsOnImage(kps_list, img.shape)
    augumented_data=[]
    for i in range(aug_num):
        image_aug, points_aug = aug(image=img, keypoints=kps)
        augumented_data.append((image_aug, points_aug))
        # todo keypoints -> np.array
        # todo keypoints -> normalize ?
    return augumented_data

def augument(inputs, aug, img, aug_num=1, split_index=50):
#     TODO: add width and height of img (change hardcoded values)
    image = img
    kps_list= []
    for idx, points in enumerate(inputs):
        start=Keypoint(x=points[0], y=points[1])
        end=Keypoint(x=points[2], y=points[3])
        kps_list.append(start)
        kps_list.append(end)

    kps= KeypointsOnImage(kps_list, image.shape)
    augumented_inputs=[]
    for i in range(aug_num):
        image_aug, points_aug = aug(image=image, keypoints=kps)
        augumentation_i=np.empty(inputs.shape)

        # split back to np.array
        for idx, start_point in enumerate(points_aug[split_index::2]):
            end_point = points_aug[idx *2 +1]
            augumentation_i[idx]=[start_point.x, start_point.y, end_point.x, end_point.y]
        print(augumentation_i[0])

        # normalize inputs
        augumentation_i[..., 0] /= image.shape[1]
        augumentation_i[..., 1] /= image.shape[0]
        augumentation_i[..., 2] /= image.shape[1]
        augumentation_i[..., 3] /= image.shape[0]
        #TODO: ^ function in train_layout
        augumented_inputs.append(augumentation_i)

    return augumented_inputs[0]#todo flatten/ parse in train layout

