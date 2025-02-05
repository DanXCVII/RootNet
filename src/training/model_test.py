import sys
import json

sys.path.append("../")

from pl_setup import MyPlSetup
from mri_dataloader import MRIDataLoader
from pytorch_lightning.loggers import TensorBoardLogger

import argparse
import torch
import pytorch_lightning as pl


def main(model_name, test_dir, threshold_eval=True):
    """
    This function loads the pytorch lightning model from the checkpoint directory and runs the test function on the test data.

    Args:
    - model_name (str): The name of the model in the run directory inside the RootNet/runs folder
    - test_dir (str): Relative path to the MRIs for testing
    - threshold_eval (bool): Whether to evaluate the model or the dice optimized thresolding
    """
    checkpoint_dir = f"../../runs/{model_name}/"
    # load the model params from the checkpoint dir (../../{model_name}/train_params.json) and store them in a dictionary
    with open(f"{checkpoint_dir}train_params.json") as f:
        train_params = json.load(f)

    my_dl = MRIDataLoader(test_dir, 1, True, 1, (96, 96, 96))

    if threshold_eval:
        save_dir = "thresholding"
    else:
        save_dir = model_name
    tb_logger = (
        TensorBoardLogger("tb_logs", name=save_dir),
    )

    model = MyPlSetup.load_from_checkpoint(f"{checkpoint_dir}best_metric_model.ckpt")
    model.threshold_eval = threshold_eval
    model.to(torch.device("cuda"))
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=tb_logger,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
        default_root_dir=".tb_logs/runs",
    )
    print("model_test.py")
    trainer.test(model=model, dataloaders=my_dl)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Test run script")

    # Add arguments
    parser.add_argument("--model_name", "-m", type=str, help="The name of the model in the run directory inside the RootNet/runs folder")
    parser.add_argument("--test_dir", "-d", type=str, help="Relative path to the MRIs for testing")
    
    # Parse the arguments
    args = parser.parse_args()

    main(
        args.model_name,
        args.test_dir
    )