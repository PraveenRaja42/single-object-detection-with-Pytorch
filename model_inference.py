import torch
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse

THRESHOLD = 0.32


def evaluate_model(img, model):
    image = utils.read_image(img)
    predictions = model.predict(image)
    labels, boxes, scores = predictions
    filtered_indices = np.where(scores > THRESHOLD)
    filtered_scores = scores[filtered_indices]
    filtered_boxes = boxes[filtered_indices]
    num_list = filtered_indices[0].tolist()
    filtered_labels = [labels[i] for i in num_list]
    return image, filtered_boxes, filtered_labels


def main():
    parser = argparse.ArgumentParser(description="Object detection")
    parser.add_argument("--img")
    parser.add_argument("--serialized_file", help="saved model file")
    args = parser.parse_args()
    model = core.Model.load(args.serialized_file, ["Connector"])

    image, boxes, filterd_labels = evaluate_model(args.img, model)
    return show_labeled_image(image, boxes, filterd_labels)


if __name__ == "__main__":
    main()
