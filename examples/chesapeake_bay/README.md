---
title: Land Cover Pixel Classification
nav_order: 70
has_children: false
parent: Zero2Neuro Examples
---

# Pixel Classification using Aerial Imagery Data

The data used in this example comes from the Chesapeake Bay area and combines high-resolution aerial imagery from the USDA National Agriculture Imagery Program (NAIP) with high-resolution land cover labels created by the Chesapeake Conservancy. The land cover dataset contains six classes:
- Water
- Tree Canopy/Forest
- Low Vegetation/Field
- Barren Land
- Impervious (Other)
- Impervious (Road)

The goal of this example is to use the aerial imagery to predict the land cover class of each pixel in one of these images.

We approach this problem as a semantic segmentation task. The model produces a classification for every pixel rather than assigning a single label to an entire image, which requires a new model architecture called a U-Net.

Note: The configuration files given in this example were meant for supercomputer usage. You can download the dataset locally however you may need to change the configuration files given.

---

## Data  

- [Source](citation.txt)
- [Online Dataset](https://source.coop/symbotic-computing-lab/chesapeake-land-cover-subset)  
The dataset includes:
- High-resolution aerial images with classes assigned to each pixel

---

## Pixel Classification

Unlike more traditional image classification where one image gets one label, this problem requires each pixel inside the image to get a label.  

The model learns the spatial patterns in the imagery to determine which class each pixel belongs to. Because neighboring pixels are often related the model also needs to use spatial information from the surrounding pixels. This makes it a semantic segmentation problem.

--- 

## Networks

We've provided a U-Net architecture for pixel level classification: 
  
  - **U-Net**
    - A variant of a convolutional neural network designed for image segmentation
    - Uses an encoder to learn the spatial features from the image
    - Uses a decoder to predict at the original image resolution
    - Uses skip connections to preserve spatial information
    - Produces a classification prediction for each pixel in the image
  
---

## Notes

This example showcases an image classification task where one image produces a large amount of predictions per image.