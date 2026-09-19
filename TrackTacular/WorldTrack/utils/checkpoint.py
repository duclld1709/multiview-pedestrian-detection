import os
import shutil
import warnings
from pathlib import Path

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint


class BestCheckpointAlias(pl.Callback):
    """Copy the monitored best checkpoint to a stable ``best.ckpt`` name."""

    def __init__(self, filename: str = "best.ckpt") -> None:
        super().__init__()
        self.filename = filename

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        checkpoint_callback = next(
            (
                callback
                for callback in trainer.callbacks
                if isinstance(callback, ModelCheckpoint) and callback.monitor == "val_center"
            ),
            None,
        )
        if checkpoint_callback is None or not checkpoint_callback.best_model_path:
            warnings.warn("No best val_center checkpoint was recorded; best.ckpt was not created.")
            return

        source = Path(checkpoint_callback.best_model_path)
        if not source.is_file():
            warnings.warn(f"Best checkpoint does not exist: {source}")
            return

        destination = source.parent / self.filename
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)

        score = checkpoint_callback.best_model_score
        metadata = source.parent / "best_checkpoint_source.txt"
        metadata.write_text(
            f"source={source.name}\nval_center={score}\n",
            encoding="utf-8",
        )
        print(f"Best checkpoint alias: {destination} (source={source.name}, val_center={score})")
