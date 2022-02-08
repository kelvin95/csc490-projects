# Module 1: Object Detection

This repository contains the starter code for Assignment 1 of CSC490H1.
In this assignment, you will implement a basic object detector for self-driving.

## How to run

### Overfitting

To overfit the detector to a single frame of PandaSet, run the following command
from the root directory of this repository:

```bash
python -m detection.main overfit --data_root=<your_path_to_dataset> --output_root=<your_path_to_outputs>
```

This command will write model checkpoints and visualizations to `<your_path_to_outputs>`.

### Training

To train the detector on the training split, run the following command
from the root directory of this repository:

```bash
python -m detection.main train --data_root=<your_path_to_dataset> --output_root=<your_path_to_outputs>
```

This command will write model checkpoints and visualizations to `<your_path_to_outputs>`.

### Visualization

To visualize the detections of the detector, run the following command
from the root directory of this repository:

```bash
python -m detection.main test --data_root=<your_path_to_dataset> --output_root=<your_path_to_outputs> --checkpoint_path<your_path_to_checkpoint>
```

This command will save detection visualizations to `<your_path_to_outputs>`.

### Evaluation

To evaluate the detections of the detector, run the following command
from the root directory of this repository:

```bash
python -m detection.main evaluate --data_root=<your_path_to_dataset> --output_root=<your_path_to_outputs> --checkpoint_path<your_path_to_checkpoint>
```

This command will save detection visualizations and metrics to `<your_path_to_outputs>`.
