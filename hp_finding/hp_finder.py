#  uv run  -m hp_finding/hp_finder.py hparams_search=cnn_layers_search_optuna experiment=find_cnn_layers
#  python hp_finding/hp_finder.py hparams_search=cnn_layers_search_optuna experiment=find_cnn_layers

import hydra
import optuna
import pyrootutils
import pytorch_lightning as pl  # noqa: E402
import torch
from omegaconf import DictConfig, OmegaConf
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import Logger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from hp_finding.tunable_model import Net  # noqa: E402
from src import utils  # noqa: E402
from src.datamodule.lung_colon_cancer_datamodule import LungColonCancerDataModule  # noqa: E402

log = utils.get_pylogger(__name__)
#  Register a resolver for torch dtypes
OmegaConf.register_new_resolver("torch_dtype", lambda name: getattr(torch, name))


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    # Load the configuration
    log.info(f"Configuration: {cfg}")
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    # Set the input shape and class names
    CLASS_NAMES = [
        "colon-adenocarcinoma",
        "colon-benign-tissue",
        "lung-adenocarcinoma",
        "lung-benign-tissue",
        "lung-squamous-cell-carcinoma",
    ]
    # sweeper_params = cfg.experiment.sweeper_params
    input_shape = [3] + hydra.utils.instantiate(cfg.data.train_preprocess_transforms[1].size)
    # Create the MLFlow logger
    # mlflow_logger = MLFlowLogger(
    #     experiment_name=cfg.logger.experiment_name,
    #     tracking_uri=cfg.logger.tracking_uri,
    #     run_name=cfg.logger.run_name,
    # )
    # logger: list[Logger] = hydra.utils.instantiate(cfg.get("logger"), experiment_name=cfg.experiments)

    def objective(trial: optuna.trial.Trial) -> float:
        # Define the hyperparameters to optimize
        total_conv_layers = trial.suggest_int("conv_layers", 1, 6)
        total_cls_hidden_layers = trial.suggest_int("hidden_layers", 1, 6)
        conv_channels = [x * 32 for x in range(1, total_conv_layers)]
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

        # Create the model
        net = Net(
            input_shape=input_shape,
            conv_layers=conv_channels,
            dropout_rate=dropout_rate,
            num_classes=len(CLASS_NAMES),
            num_hidden_layers=total_cls_hidden_layers,
        )
        # # Create the optimizer
        # optimizer = torch.optim.AdamW

        # Create the scheduler
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        model = hydra.utils.instantiate(cfg.model, model=net)
        data_module: LungColonCancerDataModule = hydra.utils.instantiate(cfg.datamodule, subset_size=0.4)  # cfg.get("datamodule")

        # Train the model
        # callbacks = [PyTorchLightningPruningCallback(trial, monitor="val_acc")]
        trainer = pl.Trainer(
            logger=True,
            accelerator="gpu",
            devices=1,
            # callbacks=callbacks,
            max_epochs=10,
            enable_progress_bar=False,
            precision=32,
            log_every_n_steps=1,
        )
        hyperparameters = {"conv_layers": conv_channels, "hidden_layers": total_cls_hidden_layers, "dropout_rate": dropout_rate}
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=data_module)

        # Evaluate the model on the validation set
        # trainer.validate(model, datamodule=datamododule)

        # return trainer.callback_metrics["val_acc"].item()

    # Run the hyperparameter optimization
    # pruner = optuna.pruners.MedianPruner()
    # study = optuna.create_study(direction="maximize", pruner=pruner)
    # study.optimize(objective, n_trials=5, timeout=600, show_progress_bar=True, n_jobs=16)t

    # # Log the best hyperparameters
    # log.info(f"Best hyperparameters: {study.best_params}")
    # log.info(f"Best trial: {study.best_trial}")


if __name__ == "__main__":
    main()
