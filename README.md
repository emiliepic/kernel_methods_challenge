# Project README

This repository contains the implementation of different machine learning models and their evaluation using cross-validation. Below is a description of the key components and how to use them.


- **dataloader.py**: This script is responsible for loading and preparing the data for the models.
  
- **kernels.py**: This script contains the implementation of the various kernels used in the machine learning algorithms.

- **log_reg.py**: Implementation of the **Logistic Regression** model.

- **krr.py**: This file contains the implementation of **Kernel Ridge Regression** and **Weighted Kernel Ridge Regression**.

- **svc.py**: This script implements the **Support Vector Classifier**.

- **solve_qud.jl**: A Julia file used to solve the quadratic optimization problem for the **Support Vector Classifier**.

- **cross_validation.py**: Implements **cross-validation** for model evaluation.

- **run.py**: This script runs cross-validation over all the kernels and models, generates a JSON file with the best parameters for each combination of kernel type and model class, and outputs results for each kernel.

- **compare_best_results.py**: This script compares the results obtained after cross-validation and identifies the best model for each dataset. You can modify the paths according to your experiments.

- **start.py**: Use this script to create the final output using the best model for each class after cross-validation. You can modify the dictionnary defining the models according to your results after running **compare_best_results.py**.

