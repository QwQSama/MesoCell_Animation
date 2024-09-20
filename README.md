# Machine Learning in Multi-scale Geomechanics
This project explores the application of machine learning techniques to simulate granular materials' behavior by replacing the traditional H-model with artificial neural networks (ANNs). The goal is to improve computational efficiency while maintaining the physical principles governing granular materials.

## Table of Contents
1. [Overview](#overview)
2. [Project Goals](#project-goals)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Future Work](#future-work)
6. [References](#references)

## Overview
Granular materials are known for their complex behavior, often requiring intensive computational resources to simulate. This project leverages machine learning to replace the traditional H-model, a micromechanical-based constitutive model, to speed up simulations and maintain accuracy in predicting granular material behavior under different stress conditions.

## Project Goals
- Replace the traditional 2D H-model framework with machine learning models.
- Generate datasets from the H-model for training and testing neural networks.
- Compare the performance of the ANN-based model with the H-model in standard geomechanical tests like isotropic compression and biaxial tests.

## Methodology
The project involves several steps:
1. **Data Generation**: Data was generated from the analytical H-model, with incremental loading in different directions and conditions.
2. **Neural Network Model**: A simple feedforward neural network with 3 hidden layers of 8 nodes each was designed to predict the deformation and forces in granular materials.
3. **Normalization Techniques**: Various data normalization techniques (e.g., Min-Max Scaling, Robust Scaling) were used to ensure training stability.
4. **Evaluation**: The trained model was tested on isotropic compression and biaxial tests, comparing the results with the traditional H-model.

## Results
The neural network model provided results similar to the traditional H-model in both isotropic compression and biaxial loading tests. 

Key findings include:
- The machine learning model replicated the H-model behavior efficiently.
- Using robust scaling yielded the best performance for predicting the behavior of granular materials.

## Future Work
- **Extend to Complex Structures**: Explore using more complex granular structures for simulation and testing.
- **Automatic Calibration**: Set up automatic calibration procedures for the machine learning model to generalize across different granular material types.


