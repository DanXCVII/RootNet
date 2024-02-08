from enum import Enum
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_setup import ModelType
from pl_setup import MyUNETRWrapper
from mri_dataloader import MRIDataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch
import json
import argparse

"""
Description:    This script avoids the problem for the training of UNETR etc.
                where the training crashes after a few epochs due to a memory error.
                It does it by training the model for several epochs, storing the model
                and restarting the training from the stored model.
Usage:  Adjust the train params (train_params) to your needs and set the model with the model variable.
        The max_epochs are set to 150 and store_model_epoch to 30 which should work for most cases. If
        the training crashes with a memory error, try to set store_model_epoch to a lower value.
Example: python single_train_run.py
"""


class MyUNETRSetup:
    def __init__(
        self,
        train_params,
        model_name,
        checkpoint_path,
        best_model_checkpoint,
        max_epochs=500,
        check_val=10,
        store_model_epoch=30,
    ):
        """
        Pipeline for training the UNETR (with superresolution) for the data provided in RootNet/data

        Args:
        - model_params: dictionary containing the parameters for the MyUNETRWrapper class
        - model_name: name of the model for creating the checkpoint dir and tensorboard logs
        - checkpoint_path: path to the checkpoint dir
        - best_model_checkpoint: name of the best model checkpoint
        - max_epochs: maximum number of epochs to train
        - check_val: number of epochs after which to perform validation
        - store_model_epoch: number of epochs after which to store the model, stop and restart training
        """
        self.max_epochs = max_epochs
        self.check_val = check_val
        self.store_model_epoch = store_model_epoch
        self.checkpoint_path = checkpoint_path
        self.best_model_checkpoint = best_model_checkpoint
        self.train_params = train_params

        self.model_name = model_name
        self._setup_checkpointing()
        self._setup_logger()
        # self.setup_profiler()

    def _setup_checkpointing(self):
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_path,
            filename=self.best_model_checkpoint,
            save_top_k=1,
            monitor="Validation/avg_val_dice",
            mode="max",
            every_n_epochs=1,
        )

    def _setup_logger(self):
        self.tb_logger = (
            TensorBoardLogger("tb_logs", name=self.model_name),
        )  # name="my_model")

    def _get_trainer(self, max_epochs):
        nnodes = os.getenv("SLURM_NNODES", 1)
        trainer = pl.Trainer(
            accelerator="gpu",
            strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
            max_epochs=max_epochs,
            check_val_every_n_epoch=self.check_val,
            logger=self.tb_logger,
            callbacks=[self.checkpoint_callback],
            gradient_clip_val=5,
            precision=16,
            log_every_n_steps=42,
            default_root_dir=".tb_logs/runs",
            # devices=4,
            # num_nodes=int(nnodes),
            # sync_batchnorm=True,
            # resume_from_checkpoint=checkpoint_path,
            # profiler=self.profiler,
        )
        print(f"trainer.max_epochs: {trainer.max_epochs}")

        return trainer

    # def _setup_profiler(self):
    #     self.profiler = PyTorchProfiler(
    #         profiled_functions=["forward", "training_step"],
    #         record_shapes=True,
    #         profile_memory=True,
    #         use_cuda=True,
    #     )

    def _get_continue_train_epoch(self, checkpoint_path):
        continue_training_epoch = 0
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            continue_training_epoch = checkpoint["epoch"]

        return continue_training_epoch

    def train(self, model_checkpoint=None):
        """
        Trains the model for store_model_epoch epochs until max_epochs is reached
        This is because a (Pin memory error arises otherwise)

        Args:
        - model_checkpoint: path to the model checkpoint to continue training from
        """
        continue_training_epoch = self._get_continue_train_epoch(model_checkpoint)
        print(f"continue_training_epoch: {continue_training_epoch}")

        for i in range(0, self.max_epochs + 1, self.store_model_epoch):
            if i > continue_training_epoch:
                max_epochs_adjusted = i
                print(f"max_epochs_adjusted: {max_epochs_adjusted}")

                trainer = self._get_trainer(max_epochs=max_epochs_adjusted)

                model = MyUNETRWrapper(
                    learning_rate=self.train_params["learning_rate"],
                    model=self.train_params["model"],
                    img_shape=self.train_params["img_shape"],
                    model_params=self.train_params["model_params"],
                )
                dm = MRIDataLoader(
                    relative_data_path=self.train_params["relative_data_path"],
                    batch_size=self.train_params["batch_size"],
                    upscale=self.train_params["upscale"],
                    samples_per_volume=self.train_params["samples_per_volume"],
                    img_shape=self.train_params["img_shape"],
                )
                if i <= self.store_model_epoch:
                    trainer.fit(model, datamodule=dm)
                else:
                    trainer.fit(model, ckpt_path=model_checkpoint, datamodule=dm)
                trainer.save_checkpoint(model_checkpoint)

    def test(self, ckpt_path=None):
        model = MyUNETRWrapper(
            learning_rate=self.train_params["learning_rate"],
            model=self.train_params["model"],
            img_shape=self.train_params["img_shape"],
            model_params=self.train_params["model_params"],
        )
        dm = MRIDataLoader(
            relative_data_path=self.train_params["relative_data_path"],
            batch_size=self.train_params["batch_size"],
            upscale=self.train_params["upscale"],
            samples_per_volume=self.train_params["samples_per_volume"],
            img_shape=self.train_params["img_shape"],
        )

        trainer = self._get_trainer(checkpoint_path=ckpt_path)

        trainer.test(model, ckpt_path=ckpt_path)


# Example Usage:

# model = UNet()

# model = MyUpscaleSwinUNETR(img_shape=img_shape, in_channels=1, out_channels=2)

# model = MyUNETR(
#     in_channels=1,
#     out_channels=2,
#     feature_size=16,
#     img_shape=(96, 96, 96),
# )

# model = UNETR()


def main(
    relative_data_path,
    batch_size,
    samples_per_volume,
    upscale,
    learning_rate,
    model,
    model_params,
    max_epochs,
    img_shape,
):
    train_params = {
        "relative_data_path": relative_data_path,
        "batch_size": batch_size,
        "upscale": upscale,
        "learning_rate": learning_rate,
        "model": model,
        "model_params": model_params,
        "samples_per_volume": samples_per_volume,
        "img_shape": img_shape,
    }

    model_name = f"weight_1_1.1_Data_DICE_sigmoid_{model}-img_shape_{img_shape[0]}-feat_{model_params['feature_size']}-upscale_{upscale}-out_channels_{model_params['out_channels']}-lr_{learning_rate}-upsample_end_{model_params['upsample_end']}"
    checkpoint_path = f"../../runs/{model_name}"
    checkpoint_file = "latest_model"
    best_checkpoint_file = "best_metric_model"

    # create the directory checkpoint dir and save train_params to a json file in it
    os.makedirs(checkpoint_path, exist_ok=True)
    with open(f"{checkpoint_path}/train_params.json", "w") as f:
        json.dump(train_params, f, indent=4)

    check_val = 10
    store_model_epoch = (
        60  # 30 seems to work safely with batch size 4 and 3 samples per volume
    )

    training_pipeline = MyUNETRSetup(
        train_params,
        model_name,
        checkpoint_path,
        best_checkpoint_file,
        max_epochs=max_epochs,
        check_val=check_val,
        store_model_epoch=store_model_epoch,
    )

    training_pipeline.train(
        model_checkpoint=f"{checkpoint_path}/{checkpoint_file}.ckpt",
    )


############## TEST ##############

# training_pipeline.test("../../runs/best_metric_model-v2.ckpt")
# training_pipeline.print_model_stats()

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Single train run script")

    # Add arguments
    parser.add_argument("--rel_data_path", "-d", type=str, help="The relative path to the data")
    parser.add_argument("--batch_size", "-b", type=int, help="An integer parameter")
    parser.add_argument("--samples_per_volume", "-spv", type=int, help="An integer parameter")
    parser.add_argument("--upscale", "-u", type=bool, help="A boolean parameter")
    parser.add_argument("--patch_size", "-p", nargs="+", type=int, help="A tuple of three integers")
    parser.add_argument("--learning_rate", "-lr", type=float, help="A float parameter for the learning rate")
    parser.add_argument("--model", "-m", type=str, help="A string parameter for the model")
    parser.add_argument("--model_params", "-mp", type=str, help="JSON String for the model parameters")
    parser.add_argument("--max_epochs", "-me", type=int, default=60, help="An integer parameter for the maximum number of epochs")

    # Parse the arguments
    args = parser.parse_args()

    try:
        model_params = json.loads(args.model_params)
    except:
        raise Exception("model_params must be a JSON string")

    main(
        args.rel_data_path,
        args.batch_size,
        args.samples_per_volume,
        args.upscale,
        args.learning_rate,
        args.model,
        model_params,
        args.max_epochs,
        args.patch_size,
    )