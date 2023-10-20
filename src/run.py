# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import hydra
import torch
from omegaconf import DictConfig
from typing import List
from pytorch_lightning import seed_everything, LightningDataModule, LightningModule, Trainer, Callback
from pytorch_lightning.loggers import LightningLoggerBase
import os


def download_checkpoint(loggers, wb_ckpt) -> None:
    if os.environ.get("LOCAL_RANK", 0) == 0:
        artifact = loggers[0].experiment.use_artifact(wb_ckpt, type="model")
        artifact_dir = artifact.download("ckpt")


@hydra.main(config_path="../configs/", config_name="run.yaml")
def main(config: DictConfig) -> None:

    seed_everything(config.seed, workers=True)
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))

    loggers: List[LightningLoggerBase] = []
    if "loggers" in config:
        for _, lg_conf in config.loggers.items():
            loggers.append(hydra.utils.instantiate(lg_conf))

    if config.resume.checkpoint is None:
        model: LightningModule = hydra.utils.instantiate(
            config.model, data_size=datamodule.tensor_size_train, _recursive_=False
        )
    else:
        download_checkpoint(loggers, config.resume.checkpoint)
        ckpt_path = "ckpt/model.ckpt"
        modelClass = hydra.utils.get_class(config.model._target_)

        model = modelClass.load_from_checkpoint(
            ckpt_path, wb_artifact=config.resume.checkpoint, **config.resume.model_overrides
        )

        if config.resume.resume_trainer and config.action == "fit":
            config.trainer.resume_from_checkpoint = ckpt_path

    # from pytorch_lightning.plugins import DDPPlugin
    # strategy = DDPPlugin(gradient_as_bucket_view=True)
    strategy = None
    if torch.cuda.device_count() > 1:
        strategy = "ddp"
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, strategy=strategy, callbacks=callbacks, logger=loggers, _convert_="partial"
    )
    if config.action == "fit":
        trainer.fit(model=model, datamodule=datamodule)
    elif config.action == "validate":
        trainer.validate(model=model, datamodule=datamodule)
    elif config.action == "test":
        trainer.test(model=model, datamodule=datamodule)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
