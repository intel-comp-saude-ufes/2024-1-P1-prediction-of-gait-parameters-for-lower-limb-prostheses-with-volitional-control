# Prediction of Gait Parameters For Lower Limb Prostheses with Volitional Control

## Overview
This repository contains code for analyzing Electromyography (EMG) signals to predict the knee joint angle in lower limb robotic prostheses. The project includes several processing and machine learning techniques to filter and extract meaningful features from EMG signals, and then use these features to predict knee joint angles.

## Table of Contents
- [Installation](#installation)
- [Data Description](#data-description)
- [Usage](#usage)
  - [Main Program](#main-program)
  - [Additional Scripts](#additional-scripts)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation
1. Clone this repository:
    ```sh
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Data Description
The data used in this project consists of EMG signals and knee joint angles. The dataset includes the following files:
- `data/train/emg_filtered.csv`: Filtered EMG signals for training.
- `data/train/torques.csv`: Torque data for training.
- `data/train/grf.csv`: Ground reaction force data for training.
- `data/train/angles.csv`: Knee joint angles for training.
- `data/test/emg_filtered.csv`: Filtered EMG signals for testing.
- `data/test/torques.csv`: Torque data for testing.
- `data/test/grf.csv`: Ground reaction force data for testing.
- `data/test/angles.csv`: Knee joint angles for testing.
- `data/1Nmar.csv`: Raw EMG and knee joint angle data.

## Usage
### Main Program
The main program trains a K-Nearest Neighbors (KNN) model to predict knee joint angles based on EMG, torque, and ground reaction force data. 
