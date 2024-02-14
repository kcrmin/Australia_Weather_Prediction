# DNN Weather Prediction

This repository contains code for predicting rainfall using Deep Neural Networks (DNNs) implemented with PyTorch. The model predicts whether it will rain tomorrow based on various weather attributes.

## Getting Started

### Prerequisites
Make sure you have the following dependencies installed:
- Python (>=3.6)
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Installation
1. Clone this repository to your local machine using:
```bash
git clone https://github.com/your-username/your-repository.git
```
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) and save it to the repository directory.

### Usage
1. Set up your environment by installing the necessary dependencies.
2. Update the `csv_path` variable in the code to point to the downloaded dataset.
3. Adjust configurations such as dataset ratios, batch size, learning rate, and epochs in the code as needed.
4. Run the code to train the DNN model and visualize the results.

## Code Structure
- **Import**: Import necessary libraries.
- **Configure**: Set up configurations such as file paths, column names, and preprocessing parameters.
- **Preprocessing**: Clean the dataset, handle missing values, encode categorical variables, and scale numerical features.
- **Load Data**: Load the preprocessed data into PyTorch DataLoader objects for training, validation, and testing.
- **DNN Model**: Define the Deep Neural Network architecture, set loss and optimizer functions, train the model, and evaluate its performance.
- **Visualization**: Visualize the model's performance using a confusion matrix.
