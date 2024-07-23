# MFDP Experiments

This repository contains the experimental setup and training scripts for the House price prediction models.

## Setup

1. Clone the repository:

   ```
   git clone [repository-url]
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

4. Set up Weights & Biases (wandb) for experiment tracking:
   ```
   wandb login
   ```

## Data Preparation

The data preparation scripts are located in the `src/` directory:

- `prepare_func.py`: Contains utility functions for data processing
- `prepare_msk.py`: Prepares data for Moscow model
- `prepare_ru.py`: Prepares data for Russia model

To prepare the data, you can use the following Makefile commands:

```
make prepare-msk
make prepare-ru
```

Or run the Python scripts directly:

```
python src/prepare_msk.py
python src/prepare_ru.py
```

## Model Training

The main training script is `src/train.py`. It uses AutoGluon to train models for both Moscow and Russia. To start training, you can use the Makefile command:

```
make train
```

Or run the Python script directly:

```
python src/train.py
```

## Experiment Tracking

This project uses Weights & Biases (wandb) for experiment tracking. All training runs, including hyperparameters, metrics, and artifacts, are logged to wandb.

Key points:

- Experiment tracking is done via wandb
- All data and models are saved as artifacts in the wandb repository

## Model Evaluation

The `src/evaluate.py` script can be used to evaluate trained models.

## Additional Information

- The `notebook.ipynb` Jupyter notebook can be used for interactive data exploration and analysis.
- The `Makefile` contains useful commands for running various parts of the project. You can view available commands by running `make` without arguments.
- The `test.py` file contains unit tests for the project.

## Requirements

See `requirements.txt` for a list of Python dependencies.

## Makefile Commands

The project includes a Makefile for convenience. Here are the available commands:

- `make train`: Run the training script
- `make prepare-msk`: Prepare data for the Moscow model
- `make prepare-ru`: Prepare data for the Russia model

You can run these commands instead of calling the Python scripts directly.
