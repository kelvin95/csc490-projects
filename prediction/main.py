import os
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot
from pylab import plt
from tqdm import tqdm

from prediction.dataset import PandasetPredDataset, custom_collate
from prediction.metrics.evaluator import Evaluator
from prediction.model import PredictionModel, PredictionModelConfig
from prediction.modules.loss_function import PredictionLossFunction
from prediction.utils.viz import vis_pred_labels

torch.multiprocessing.set_sharing_strategy("file_system")


def overfit(
    data_root: str,
    output_root: str,
    seed: int = 42,
    num_iterations: int = 500,
    log_frequency: int = 100,
    learning_rate: float = 1e-2,
) -> None:
    """Overfit predictor to one frame of the Pandaset dataset.

    Args:
        data_root: The root directory of the Pandaset dataset.
        output_root: The root directory to output visualizations and checkpoints.
        seed: A fixed random seed for reproducibility.
        num_iterations: The number of iterations to run overfitting for.
        log_frequency: The number of training iterations between logs/visualizations.
        learning_rate: The learning rate for training the model.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.random.manual_seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_root, exist_ok=True)

    # setup model
    model_config = PredictionModelConfig()
    model = PredictionModel(model_config).to(device)

    # setup data
    dataset = PandasetPredDataset(data_root, test=False)
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=custom_collate)

    # setup loss function and optimizer
    loss_fn = PredictionLossFunction(model_config.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses_buffer = defaultdict(lambda: [])
    # start training
    history_tensors, labels_tensors, labels = next(iter(dataloader))
    history_tensors = [history_tensor.to(device) for history_tensor in history_tensors]
    labels_tensors = [label_tensor.to(device) for label_tensor in labels_tensors]
    for idx in tqdm(range(num_iterations)):
        model.train()
        predictions = model(history_tensors)
        loss, loss_metadata = loss_fn(predictions, labels_tensors)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_buffer["Loss"].append(loss.item())
        losses_buffer["L1 Loss"].append(loss_metadata.l1_loss.item())
        # inference on the training example, and save vis results
        if (idx + 1) % log_frequency == 0:
            print(
                f"[{idx}/{num_iterations}]: "
                " ".join(
                    [
                        f"{key} - {sum(value) / len(value):.4f}"
                        for key, value in losses_buffer.items()
                    ]
                )
            )
            losses_buffer.clear()
            # visualize predictions and ground truth
            with torch.no_grad():
                model.eval()
                predictions = model.inference(history_tensors[0].to(device)).to("cpu")
            # We copy over the ground truth yaw and boxes for simplicity
            predictions.yaws = labels[0].yaws
            predictions.boxes = labels[0].boxes
            vis_pred_labels(predictions, labels[0])
            plt.savefig(f"{output_root}/predictions.png")
            plt.close("all")


def train(
    data_root: str,
    output_root: str,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 8,
    num_epochs: int = 25,
    log_frequency: int = int(269 // 16),
    learning_rate: float = 1e-4,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Train detector on the Pandaset dataset.

    Args:
        data_root: The root directory of the Pandaset dataset.
        output_root: The root directory to output visualizations and checkpoints.
        seed: A fixed random seed for reproducibility.
        batch_size: The batch size per training iteration.
        num_workers: The number of dataloader workers.
        num_epochs: The number of epochs to run training over.
        log_frequency: The number of training iterations between logs/visualizations.
        learning_rate: The learning rate for training the detection model.
        checkpoint_path: Optionally, whether to initialize the model from a checkpoint.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.random.manual_seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_root, exist_ok=True)

    # setup model
    model_config = PredictionModelConfig()
    model = PredictionModel(model_config)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)

    # setup train data
    train_dataset = PandasetPredDataset(data_root, test=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )

    # setup data
    test_dataset = PandasetPredDataset(data_root, test=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, num_workers=num_workers, collate_fn=custom_collate
    )

    # setup loss function and optimizer
    loss_fn = PredictionLossFunction(model_config.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate * batch_size)
    # Learning rate decay coefficient
    # Set to < 1 if you want to decay the learning rate each epoch
    lr_decay_rate = 1
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=lr_decay_rate
    )
    # These data structures are for plotting the learning progress and evaluation
    losses = [[], []]
    smoothed_losses = [[], []]
    eval_losses = [[], []]
    step = 0
    smoothed_loss = 0
    smoothing_coefficient = 0.05
    losses_buffer = defaultdict(lambda: [])

    # start training
    for epoch in range(num_epochs):
        for idx, (history_tensors, labels_tensors, labels) in tqdm(
            enumerate(train_dataloader)
        ):
            history_tensors = [
                history_tensor.to(device) for history_tensor in history_tensors
            ]
            labels_tensors = [
                label_tensor.to(device) for label_tensor in labels_tensors
            ]

            model.train()
            predictions = model(history_tensors)
            loss, loss_metadata = loss_fn(predictions, labels_tensors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses[0].append(loss.item())
            losses[1].append(step)
            if step == 0:
                smoothed_loss = loss.item()
            else:
                smoothed_loss = (
                    smoothed_loss * (1 - smoothing_coefficient)
                    + smoothing_coefficient * loss.item()
                )

            smoothed_losses[0].append(smoothed_loss)
            smoothed_losses[1].append(step)
            step += 1

            losses_buffer["Loss"].append(loss.item())
            losses_buffer["L1 Loss"].append(loss_metadata.l1_loss.item())
            # inference on the training example, and save vis results
            if (idx + 1) % log_frequency == 0:
                print(
                    f"Epoch {epoch} [{idx}/{len(train_dataloader)}]:\t",
                    "\t".join(
                        [
                            f"{key} - {sum(value) / len(value):.4f}"
                            for key, value in losses_buffer.items()
                        ]
                    ),
                    "\n",
                )
                losses_buffer.clear()

                # visualize predictions and ground truth
                with torch.no_grad():
                    model.eval()
                    predictions = model.inference(history_tensors[0].to(device)).to(
                        "cpu"
                    )
                # We copy over the ground truth yaw and boxes for simplicity
                predictions.yaws = labels[0].yaws
                predictions.boxes = labels[0].boxes
                vis_pred_labels(predictions, labels[0])
                plt.savefig(f"{output_root}/predictions.png")
                plt.close("all")

        torch.save(model.state_dict(), f"{output_root}/{epoch:03d}.pth")
        scheduler.step()

        # Get test metrics
        evaluator = Evaluator()
        for _, (history_tensors, _, labels) in tqdm(enumerate(test_dataloader)):
            model.eval()
            predictions = model.inference(history_tensors[0].to(device))
            evaluator.append(predictions.to(torch.device("cpu")), labels[0])

        result = evaluator.evaluate()
        print("------------\nTest metrics\n------------")
        print(result)
        print("------------")
        eval_losses[0].append(result["ADE"][0])
        eval_losses[1].append(step)

    pyplot.plot(losses[1], losses[0], label="Training Loss (L1)")
    pyplot.plot(
        smoothed_losses[1],
        smoothed_losses[0],
        "r--",
        label="Smoothed Training Loss (L1)",
    )
    pyplot.xlabel("step")
    pyplot.ylabel("loss")
    pyplot.legend()
    plt.savefig(f"{output_root}/training_loss.png")
    pyplot.close("all")
    pyplot.plot(eval_losses[1], eval_losses[0], label="Eval ADE")
    pyplot.xlabel("step")
    pyplot.ylabel("ADE")
    pyplot.legend()
    plt.savefig(f"{output_root}/evaluation_loss.png")


@torch.no_grad()
def test(
    data_root: str,
    output_root: str,
    seed: int = 42,
    num_workers: int = 8,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Visualize the outputs of the detector on Pandaset.

    Args:
        data_root: The root directory of the Pandaset dataset.
        output_root: The root directory to output visualizations and checkpoints.
        seed: A fixed random seed for reproducibility.
        num_workers: The number of dataloader workers.
        checkpoint_path: Optionally, whether to initialize the model from a checkpoint.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.random.manual_seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_root, exist_ok=True)

    # setup model
    model_config = PredictionModelConfig()
    model = PredictionModel(model_config)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)

    # setup data
    dataset = PandasetPredDataset(data_root, test=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, collate_fn=custom_collate
    )

    for idx, (history_tensors, _, labels) in tqdm(enumerate(dataloader)):
        model.eval()
        history_tensors = [
            history_tensor.to(device) for history_tensor in history_tensors
        ]
        predictions = model.inference(history_tensors[0].to(device)).to("cpu")
        # We copy over the ground truth yaw and boxes for simplicity
        predictions.yaws = labels[0].yaws
        predictions.boxes = labels[0].boxes
        vis_pred_labels(predictions, labels[0])
        plt.savefig(f"{output_root}/{idx:03d}.png")
        plt.close("all")


@torch.no_grad()
def evaluate(
    data_root: str,
    output_root: str,
    seed: int = 42,
    num_workers: int = 8,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Evaluate the detector on Pandaset and save its metrics.

    Args:
        data_root: The root directory of the Pandaset dataset.
        output_root: The root directory to output visualizations and checkpoints.
        seed: A fixed random seed for reproducibility.
        num_workers: The number of dataloader workers.
        checkpoint_path: Optionally, whether to initialize the model from a checkpoint.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.random.manual_seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_root, exist_ok=True)

    # setup model
    model_config = PredictionModelConfig()
    model = PredictionModel(model_config)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)

    # setup data
    dataset = PandasetPredDataset(data_root, test=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, collate_fn=custom_collate
    )

    evaluator = Evaluator()
    for _, (history_tensors, _, labels) in tqdm(enumerate(dataloader)):
        model.eval()
        predictions = model.inference(history_tensors[0].to(device))
        evaluator.append(predictions.to(torch.device("cpu")), labels[0])

    result = evaluator.evaluate()
    with open(f"{output_root}/result.csv", "w") as f:
        f.write(result.to_csv())

    evaluator.eval_visualize(output_root)
    plt.close("all")

    print(result)


if __name__ == "__main__":
    import fire

    fire.Fire()
