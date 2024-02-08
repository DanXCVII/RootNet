from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray import tune
import ray
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from pl_setup import MyUNETRWrapper
from mri_dataloader import MRIDataLoader
import pytorch_lightning as pl
from ray.tune.schedulers import ASHAScheduler
import os


class MyUNETray:
    def __init__(self):
        self.search_space = {
            "rel_data_path": "../../data",
            "batch_size": tune.choice([3, 2]),
            "upscale": True,
            "learning_rate": tune.choice([0.0003, 0.0005, 0.0008]),
            "model": tune.choice([ModelType.UPSCALESWINUNETR.name]),
            "model_params": {
                "in_channels": 1,
                "out_channels": 2,
                "feature_size": tune.choice([24]),
            },
            "samples_per_volume": 3,
            "img_shape": tune.choice([(96, 96, 96)]),
        }

        scaling_config = ScalingConfig(
            num_workers=2,
            use_gpu=True,
            resources_per_worker={"CPU": 64, "GPU": 4},
        )

        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="Validation/avg_val_dice",
                checkpoint_score_order="max",
            ),
        )

        # Define a TorchTrainer without hyper-parameters for Tuner
        self.ray_trainer = TorchTrainer(
            self.train_func,
            scaling_config=scaling_config,
            run_config=run_config,
        )

    def train_func(self, config):
        print("samples_per_volume actual: ", config["samples_per_volume"])

        self.dm = MRIDataLoader(
            relative_data_path=config["rel_data_path"],
            batch_size=config["batch_size"],
            upscale=config["upscale"],
            samples_per_volume=config["samples_per_volume"],
            img_shape=config["img_shape"],
        )

        self.model = MyUNETRWrapper(
            learning_rate=config["learning_rate"],
            model=config["model"],
            model_params=config["model_params"],
            img_shape=config["img_shape"],
        )

        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(self.model, datamodule=self.dm)

    def tune_mnist_asha(self, num_samples=10):
        scheduler = ASHAScheduler(max_t=150, grace_period=50, reduction_factor=2)

        tuner = tune.Tuner(
            self.ray_trainer,
            param_space={"train_loop_config": self.search_space},
            tune_config=tune.TuneConfig(
                metric="Validation/avg_val_dice",
                mode="max",
                num_samples=num_samples,
                scheduler=scheduler,
            ),
        )
        print("Tuning...")
        result = tuner.fit()
        print("Done!")

        return result


ray.init(address=os.environ["ip_head"], _node_ip_address=os.environ["head_node_ip"])

myUNETray = MyUNETray()
myUNETray.tune_mnist_asha()
