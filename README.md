# JAX Machine Learning Algorithms ⚡

![JAX](https://img.shields.io/badge/JAX-Accelerated-blue?style=for-the-badge&logo=google)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow?style=for-the-badge&logo=python)

A collection of fundamental Machine Learning algorithms implemented using **JAX**, **Flax**, and **Optax**. 

The goal of this repository is to demonstrate how to leverage JAX's Just-In-Time (JIT) compilation, automatic vectorization, and automatic differentiation to build efficient, scalable ML models from the ground up.

## 📂 Repository Structure

Each algorithm is contained within its own directory, featuring a Jupyter Notebook with a complete end-to-end example (data loading, preprocessing, model definition, training, and evaluation).

```text
jax-ml-algorithms/
│
├── logistic_regression/          
│   ├── flax_linen_logistic_regression.ipynb  
│
├── xgboost/               
│   └── jax_xgboost.ipynb         # Solving kaggle competition for house prices
│
├── neural_networks/              # 🚧 Coming Soon
│   └── ...
│
├── requirements.txt              # Dependencies
└── README.md                     # Main documentation