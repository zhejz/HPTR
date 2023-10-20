# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional
from pathlib import Path
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    return None
    # raise Exception("You are using wandb related callback, but WandbLogger was not found for some reason...")


class ModelCheckpointWB(ModelCheckpoint):
    def save_checkpoint(self, trainer) -> None:
        super().save_checkpoint(trainer)
        if not hasattr(self, "_logged_model_time"):
            self._logged_model_time = {}
        logger = get_wandb_logger(trainer)
        if self.current_score is None:
            self.current_score = trainer.callback_metrics.get(self.monitor)
        if logger is not None:
            self._scan_and_log_checkpoints(logger)

    @rank_zero_only
    def _scan_and_log_checkpoints(self, wb_logger: WandbLogger) -> None:
        # adapted from pytorch_lightning 1.4.0: loggers/wandb.py
        checkpoints = {
            self.last_model_path: self.current_score,
            self.best_model_path: self.best_model_score,
        }
        checkpoints = sorted((Path(p).stat().st_mtime, p, s) for p, s in checkpoints.items() if Path(p).is_file())
        checkpoints = [
            c for c in checkpoints if c[1] not in self._logged_model_time.keys() or self._logged_model_time[c[1]] < c[0]
        ]
        # log iteratively all new checkpoints
        for t, p, s in checkpoints:
            metadata = {
                "score": s.item(),
                "original_filename": Path(p).name,
                "ModelCheckpoint": {
                    k: getattr(self, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                        "_every_n_val_epochs",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(self, k)
                },
            }
            artifact = wandb.Artifact(name=wb_logger.experiment.id, type="model", metadata=metadata)
            artifact.add_file(p, name="model.ckpt")
            aliases = ["latest", "best"] if p == self.best_model_path else ["latest"]
            wb_logger.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self._log = log
        self._log_freq = log_freq

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer)
        logger.watch(model=trainer.model, log=self._log, log_freq=self._log_freq)
