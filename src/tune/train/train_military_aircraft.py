from ray import air, tune
from ray_lightning import RayPlugin
from ray_lightning.tune import TuneReportCallback
import pytorch_lightning as pl

from src.config.config_loader import ConfigLoader
from src.model.model_loader import ModelLoader
from src.model.lightning_modules.object_detection_module import ObjectDetectionModule

def train(config_path):
    config = ConfigLoader.load(config_path)
    model = ModelLoader.load(model_type=self.config.model_type)
    model = ObjectDetectionModule(model=model, lr=self.config.learning_rate)
    # Create your PTL model.
    model = MNISTClassifier(config)

    # Create the Tune Reporting Callback
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]

    trainer = pl.Trainer(
        max_epochs=4,
        callbacks=callbacks,
        plugins=[RayPlugin(num_workers=4, use_gpu=False)])
    trainer.fit(model)

    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    # Make sure to specify how many actors each training run will create via the "extra_cpu" field.
    tuner = tune.Tuner(
        tune.with_resources(train_mnist, {"cpu": 1, "extra_cpu": 4}),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=num_samples,
        ),
        param_space=config
    )

    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)