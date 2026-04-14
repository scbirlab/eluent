from typing import Mapping, Optional, Union
from argparse import Namespace
from collections import defaultdict
import os

from carabiner import pprint_dict, print_err
from carabiner.cliutils import clicommand

from .eval import _evaluate_modelbox_and_save_metrics
from .utils import _dict_to_pandas, _init_modelbox, _overwrite_config
from ..checkpoint_utils import _load_json, save_json

STRUCTURE_COLUMN_DEFAULT: str = "smiles"

def _load_modelbox_training_data(
    modelbox,
    checkpoint: Optional = None,
    cache: Optional[str] = None,
    **overrides
):
    if any([
        overrides.get("training") is not None,  # override checkpoint training data
        checkpoint is None,  # no checkpoint
        modelbox.training_data is None,  # checkpoint without training data
    ]):
        load_data_args = {
            "data": overrides.get("training"), 
            "cache": cache,
            # command-line takes precedent:
            "features": overrides.get("features"),
            "labels": overrides.get("labels"),
            "context": overrides.get("context"),
        }
        if hasattr(modelbox, "tanimoto_column"):  # i.e., is for chemistry
            # command-line takes precedent:
            load_data_args["structure_column"] = overrides.get("structure") or modelbox._default_preprocessing_args.get("structure_column")
            if load_data_args["structure_column"] is None:
                print_err(f"Structure column not provided, falling back to {STRUCTURE_COLUMN_DEFAULT}.")
                load_data_args["structure_column"] = STRUCTURE_COLUMN_DEFAULT
        pprint_dict(
            load_data_args,
            message="Data-loading configuration",
        )
        modelbox.load_training_data(**load_data_args)
    return modelbox, load_data_args


def _init_modelbox_and_load_training_data(
    cli_config: Mapping[str, Union[str, int, float]],
    checkpoint: Optional[str] = None,
    config_file: Optional[str] = None,
    config_idx: int = 0,
    cache: Optional[str] = None,
    **overrides
):
    modelbox = _init_modelbox(
        cli_config=cli_config,
        checkpoint=checkpoint,
        config_file=config_file,
        config_idx=config_idx,
        cache=cache,
        **overrides,
    )

    cli_config = _overwrite_config(
        cli_config, 
        config_file=config_file, 
        config_idx=config_idx,
    )
    modelbox, load_data_args = _load_modelbox_training_data(
        modelbox=modelbox,
        checkpoint=checkpoint,
        cache=cache,
        **(overrides | cli_config),
    )

    if checkpoint is None:  # model not instantiated yet
        modelbox.model = modelbox.create_model()
    return modelbox, load_data_args


def _train_and_save_modelbox(
    modelbox,
    early_stopping: Optional[int] = None,
    output_name: Optional[str] = None,
    **training_args
):
    from datasets.fingerprint import Hasher
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

    from ..utils.lightning import _get_most_recent_lightning_log
    from ..utils.plotting import _plot_history

    if early_stopping is not None:
        callbacks = [EarlyStopping('val_loss', patience=early_stopping)]
    else:
        callbacks = None

    if output_name is None:
        output_name = (
            f"{modelbox.class_name}_n{modelbox.size}_"
            f"y={'-'.join(modelbox._label_cols)}_"
            f"h={Hasher.hash(modelbox._input_featurizers)}",
        )
    
    checkpoint_path = output_name
    modelbox.train(
        callbacks=callbacks,
        trainer_opts={  # passed to lightning.Trainer()
            "logger": [
                CSVLogger(save_dir=os.path.join(checkpoint_path, "logs-csv")),
                TensorBoardLogger(save_dir=os.path.join(checkpoint_path, "logs")),
            ], 
            "enable_progress_bar": True, 
            "enable_model_summary": True,
        },
        **training_args,
    )

    # get latest CSV log
    max_version = _get_most_recent_lightning_log(
        os.path.join(checkpoint_path, "logs-csv"),
        "metrics.csv",
    )
    _plot_history(
        max_version,
        os.path.join(checkpoint_path, "training-log")
    )
    modelbox.save_checkpoint(checkpoint_path)
        
    return checkpoint_path, training_args


@clicommand(message='Training a Pytorch model')
def _train(args: Namespace) -> None:

    from ..autoclass import AutoModelBox

    cli_config = {
        "class_name": args.model_class.casefold(),
        "merge_method": args.fusion.casefold(),
        "use_2d": args.descriptors,
        "use_fp": args.fp,
        "n_hidden": args.hidden,
        "residual_depth": args.residual,
        "n_units": args.units,
        "dropout": args.dropout,
        "ensemble_size": args.ensemble_size,
        "learning_rate": args.learning_rate,
    }
    if args.features is None:
        if args.x2 is None:
            features = None
        else:
            features = [args.x2]
    else:
        if args.x2 is None:
            features = [args.features]
        else:
            features = [args.features, args.x2]

    if (args.x2 is not None or args.context is not None) and not args.model_class.casefold().startswith("bilinear"):
        print_err("[WARN] Can only use --x2 or --context with --model-class bilinear; concatentating to -x")
        features = [[_f for f in features for _f in f]]

    modelbox, load_data_args = _init_modelbox_and_load_training_data(
        cli_config=cli_config,
        checkpoint=args.checkpoint,
        config_file=args.config,
        config_idx=args.config_index,
        cache=args.cache,
        # overrides:
        training=args.training,
        structure=args.structure,
        structure_representation=args.input_representation,
        labels=args.labels,
        features=features,
        context=args.context,
    )

    pprint_dict(
        modelbox._model_config, 
        message=f"Initialized model {modelbox.class_name} with {modelbox.size} parameters",
    )
    
    training_args = {
        "epochs": args.epochs, 
        "batch_size": args.batch,
        "val_data": args.validation,
        "early_stopping": args.early_stopping,
    }
    pprint_dict(
        training_args, 
        message=f">> Training {modelbox.class_name} with training configuration",
    )
    checkpoint_path, training_args = _train_and_save_modelbox(
        modelbox=modelbox,
        early_stopping=args.early_stopping,
        epochs=args.epochs, 
        batch_size=args.batch,
        val_data=args.validation,
        output_name=args.output,
    )
    for obj, f in zip((training_args, load_data_args), ("training-args.json", "load-data-args.json")):
        save_json(obj, os.path.join(checkpoint_path, f))

    # Reload - built-in test that the checkpointing works!
    modelbox = AutoModelBox.from_pretrained(
        checkpoint_path, 
        cache=args.cache,
    )
    overall_metrics = defaultdict(list)
    for name in ("training", "validation", "test"):
        dataset = getattr(args, name)
        if dataset is not None:  # skip optional extra datasets, e.g. "test"
            if name == "training":
                dataset = None  # Use cached training data
            metrics = _evaluate_modelbox_and_save_metrics(
                modelbox,
                metric_filename=os.path.join(checkpoint_path, f"eval-metrics_{name}.json"),
                plot_filename=os.path.join(checkpoint_path, f"predictions_{name}"),
                dataset=dataset,
            )
            pprint_dict(
                metrics, 
                message=f"Evaluation: {name}",
            )

            overall_metrics["split"].append(name)
            overall_metrics["split_filename"].append(dataset or load_data_args["data"])
            if args.config is not None:
                overall_metrics["config_i"].append(args.config_index)
            keys_added = []
            for d in (
                modelbox._init_kwargs, 
                modelbox._model_config, 
                load_data_args, 
                training_args, 
                metrics,
            ):
                for key, val in d.items():
                    if key != "trainer_opts" and key not in keys_added:
                        overall_metrics[key].append(val)
                        keys_added.append(key)

    _dict_to_pandas(overall_metrics, os.path.join(checkpoint_path, "metrics.csv"))

    return None
