# Cat–Dog Image Classification using Convolutional Neural Networks (CNN)

## Overview

This project implements a Convolutional Neural Network (CNN)–based image classification system to distinguish between cats and dogs. The work was carried out as part of an Artificial Neural Networks (ANN) course project and developed iteratively over approximately two months, covering data preprocessing, model design, training, evaluation, and reporting.

## Dataset

* The model was trained on a binary image dataset (cats vs dogs).
* Images are preprocessed through resizing, normalization, and batching.
* Data augmentation techniques (e.g., flipping, rotation) were applied to improve generalization.

## Model & Methodology

* Implemented using PyTorch
* CNN architecture with convolutional layers for feature extraction, pooling layers for dimensionality reduction and fully connected layers for classification.
* Training includes loss monitoring, accuracy evaluation, and validation set testing.
* Performance metrics: Accuracy, Confusion matrix, Precision / recall.

## How to Run
* Clone repo `git clone https://github.com/vkvalli/cat-dog-classifier.git` `cd cat-dog-classifier`
* Run the script `python cnn_cat_prediction.py`

## Notes
* Make sure to update dataset paths before running.
* The notebook was originally developed in a cloud environment and later cleaned for reproducibility.
* Paths and environment-specific code were adjusted for local execution
* Further details, experiments, and results are documented in CNN Report.pdf
* The dataset is not included in this repository. You should provide your own dataset and update the data paths accordingly.


