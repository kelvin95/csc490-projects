import os
import random
from typing import Optional

import numpy as np
import torch
from pylab import plt
from tqdm import tqdm

from detection.dataset import PandasetDataset, custom_collate
from detection.metrics.evaluator import Evaluator
from detection.model import DetectionModel, DetectionModelConfig
from detection.modules.loss_function import DetectionLossFunction
from detection.utils.visualization import visualize_detections


def overfit(
    data_root: str,
    output_root: str,
    seed: int = 42,
    num_iterations: int = 500,
    log_frequency: int = 100,
    learning_rate: float = 1e-4,
) -> None:
    """Overfit detector to one frame of the Pandaset dataset.

    Args:
        data_root: The root directory of the Pandaset dataset.
        output_root: The root directory to output visualizations and checkpoints.
        seed: A fixed random seed for reproducibility.
        num_iterations: The number of iterations to run overfitting for.
        log_frequency: The number of training iterations between logs/visualizations.
        learning_rate: The learning rate for training the detection model.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.random.manual_seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_root, exist_ok=True)

    # setup model
    model_config = DetectionModelConfig()
    model = DetectionModel(model_config).to(device)

    # setup data
    dataset = PandasetDataset(data_root, model_config)
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=custom_collate)

    # setup loss function and optimizer
    loss_fn = DetectionLossFunction(model_config.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start training
    bev_lidar, bev_targets, labels = next(iter(dataloader))
    bev_lidar = bev_lidar.to(device)
    bev_targets = bev_targets.to(device)
    for idx in tqdm(range(num_iterations)):
        model.train()
        predictions = model(bev_lidar)
        loss, loss_metadata = loss_fn(predictions, bev_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # inference on the training example, and save vis results
        if (idx + 1) % log_frequency == 0:
            print(
                f"[{idx}/{num_iterations}]: "
                f"Loss - {loss.item():.4f} "
                f"Heatmap Loss - {loss_metadata.heatmap_loss.item():.4f} "
                f"Offset Loss - {loss_metadata.offset_loss.item():.4f} "
                f"Size Loss - {loss_metadata.size_loss.item():.4f} "
                f"Heading Loss - {loss_metadata.heading_loss.item():.4f} "
            )

            # visualize target heatmap
            target_heatmap = bev_targets[0, 0].cpu().detach().numpy()
            plt.matshow(target_heatmap, origin="lower")
            plt.savefig(f"{output_root}/target_heatmap.png")

            # visualize predicted heatmap
            predicted_heatmap = predictions[0, 0].cpu().detach().sigmoid().numpy()
            plt.matshow(predicted_heatmap, origin="lower")
            plt.savefig(f"{output_root}/predicted_heatmap.png")

            # visualize detections and ground truth
            with torch.no_grad():
                model.eval()
                detections = model.inference(bev_lidar[0].to(device))
            lidar = bev_lidar[0].sum(0).nonzero().detach().cpu()[:, [1, 0]]
            visualize_detections(lidar, detections, labels[0])
            plt.savefig(f"{output_root}/detections.png")
            plt.close("all")


def train(
    data_root: str,
    output_root: str,
    seed: int = 42,
    batch_size: int = 2,
    num_workers: int = 8,
    num_epochs: int = 5,
    log_frequency: int = 100,
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
    model_config = DetectionModelConfig()
    model = DetectionModel(model_config)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)

    # setup data
    dataset = PandasetDataset(data_root, model_config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )

    # setup loss function and optimizer
    loss_fn = DetectionLossFunction(model_config.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start training
    for epoch in range(num_epochs):
        for idx, (bev_lidar, bev_targets, labels) in tqdm(enumerate(dataloader)):
            model.train()
            predictions = model(bev_lidar.to(device))
            loss, loss_metadata = loss_fn(predictions, bev_targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # inference on the training example, and save vis results
            if (idx + 1) % log_frequency == 0:
                print(
                    f"Epoch {epoch} [{idx}/{len(dataloader)}]: "
                    f"Loss - {loss.item():.4f} "
                    f"Heatmap Loss - {loss_metadata.heatmap_loss.item():.4f} "
                    f"Offset Loss - {loss_metadata.offset_loss.item():.4f} "
                    f"Size Loss - {loss_metadata.size_loss.item():.4f} "
                    f"Heading Loss - {loss_metadata.heading_loss.item():.4f} "
                )

                # visualize target heatmap
                target_heatmap = bev_targets[0, 0].cpu().detach().numpy()
                plt.matshow(target_heatmap, origin="lower")
                plt.savefig(f"{output_root}/target_heatmap.png")

                # visualize predicted heatmap
                predicted_heatmap = predictions[0, 0].cpu().detach().sigmoid().numpy()
                plt.matshow(predicted_heatmap, origin="lower")
                plt.savefig(f"{output_root}/predicted_heatmap.png")

                # visualize detections and ground truth
                with torch.no_grad():
                    model.eval()
                    detections = model.inference(bev_lidar[0].to(device))
                lidar = bev_lidar[0].sum(0).nonzero().detach().cpu()[:, [1, 0]]
                visualize_detections(lidar, detections, labels[0])
                plt.savefig(f"{output_root}/detections.png")
                plt.close("all")

        torch.save(model.state_dict(), f"{output_root}/{epoch:03d}.pth")


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
    model_config = DetectionModelConfig()
    model = DetectionModel(model_config)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)

    # setup data
    dataset = PandasetDataset(data_root, model_config, test=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, collate_fn=custom_collate
    )

    for idx, (bev_lidar, _, labels) in tqdm(enumerate(dataloader)):
        model.eval()
        detections = model.inference(bev_lidar[0].to(device))
        lidar = bev_lidar[0].sum(0).nonzero().detach().cpu()[:, [1, 0]]
        visualize_detections(lidar, detections, labels[0])
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
    model_config = DetectionModelConfig()
    model = DetectionModel(model_config)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)

    # setup data
    dataset = PandasetDataset(data_root, model_config, test=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, collate_fn=custom_collate
    )

    evaluator = Evaluator(ap_thresholds=[2.0, 4.0, 8.0, 16.0])
    for _, (bev_lidar, _, labels) in tqdm(enumerate(dataloader)):
        model.eval()
        detections = model.inference(bev_lidar[0].to(device))
        evaluator.append(detections.to(torch.device("cpu")), labels[0])

    result = evaluator.evaluate()
    result_df = result.as_dataframe()
    with open(f"{output_root}/result.csv", "w") as f:
        f.write(result_df.to_csv())

    result.visualize()
    plt.savefig(f"{output_root}/results.png")
    plt.close("all")

    print(result_df)


if __name__ == "__main__":
    import fire

    fire.Fire()
