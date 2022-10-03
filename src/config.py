# file config.py
# author Pavel Ševčík

import logging
import functools

from .dataset import GraphDataset, KnnRectangeCenterBuild
from .dataset import AddVectorAttr, SequentialDataBuild
from .model import MultiHeadModel, model_factory

class Config:
    def configure(self, data_config, model_config):
        """Returns a function that takes csv_path as a single argument and a configured model"""
        model = model_factory(model_config)
        data_build = self._get_data_build(model_config, model)
        graph_build = self._get_graph_build(data_config["graph_build"])
        kwargs = {
            "graph_features": data_config["graph_features"],
            "data_build": data_build,
            "graph_build": graph_build
        }
        dataset_builder = functools.partial(GraphDataset, **kwargs)
        return dataset_builder, model

    def _get_graph_build(self, config):
        type = config["type"].lower()
        del config["type"]
        if type == "knn_rectangle_center":
            return KnnRectangeCenterBuild(**config)
        else:
            msg = f"Unknown graph build type '{type}'"
            logging.error(msg)
            raise ValueError(msg)

    def _get_data_build(self, config, model: MultiHeadModel):
        data_builds = []

        input_transforms_config = config["input_transforms"]
        for input_transform in input_transforms_config:
            input_type = input_transform["type"].lower()
            del input_transform["type"]
            if input_type == "vector":
                data_builds.append(AddVectorAttr(**input_transform))
            else:
                msg = f"Unknown input type '{input_type}'"
                logging.error(msg)
                raise ValueError(msg)

        head_data_builds = [head.get_data_build() for head in model.heads]
        data_builds.extend(head_data_builds)

        return SequentialDataBuild(data_builds)
            

