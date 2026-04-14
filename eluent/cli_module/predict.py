from typing import Mapping, Optional, Union

from argparse import Namespace
from collections import defaultdict
import os

from carabiner import cast, pprint_dict, print_err
from carabiner.cliutils import clicommand

from .eval import _evaluate_modelbox_and_save_metrics
from .io import _resolve_and_slice_data, _save_dataset

from .utils import _dict_to_pandas


@clicommand("Predicting with the following parameters")
def _predict(args: Namespace) -> None:

    import torch
    from ..autoclass import AutoModelBox

    
    output = args.output
    out_dir = os.path.dirname(output)
    if len(out_dir) > 0 and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    common_args = {
        "batch_size": args.batch,
        "cache": args.cache,
    }
    candidates_ds = _resolve_and_slice_data(
        args.test,
        start=args.start,
        end=args.end,
    )
    modelbox = AutoModelBox.from_pretrained(
        args.checkpoint, 
        cache=args.cache,
    )
    if hasattr(modelbox, "tanimoto_nn"):
        preprocessing_args = {
            "structure_column": args.structure,
            "input_representation": args.input_representation,
        }
    else:
        preprocessing_args = {}

    pprint_dict(
        modelbox._model_config, 
        message=f"Initialized model {modelbox.class_name} with {modelbox.size} parameters",
    )
    for col in cast(modelbox._label_cols, to=list):
        if col not in candidates_ds.column_names:
            from numpy import zeros_like
            candidates_ds = candidates_ds.add_column(
                col,
                zeros_like(
                    candidates_ds
                    .with_format("numpy")
                    [candidates_ds.column_names[0]]
                )
            )
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
    if (args.x2 is not None or args.context is not None) and not modelbox.class_name.startswith("bilinear"):
        print_err("[WARN] Can only use --x2 or --context with --model-class bilinear; concatentating to -x")
        features = [[_f for f in features for _f in f]]
    context = args.context
    
    preprocessing_args["_extra_cols_to_keep"] = (args.extras or [])
    modelbox.to("cuda" if torch.cuda.is_available() else "cpu")
    candidates_ds = modelbox.predict(
        data=candidates_ds,
        aggregator="mean",
        features=features,
        context=context,
        **preprocessing_args,
        **common_args,
    )
    preprocessing_args["_extra_cols_to_keep"].append(modelbox._prediction_key)
    if args.variance:
        candidates_ds = modelbox.prediction_variance(
            candidates=candidates_ds,
            features=features,
            context=context,
            **preprocessing_args,
            **common_args,
        )
        preprocessing_args["_extra_cols_to_keep"].append(modelbox._variance_key)
    if args.tanimoto:
        if hasattr(modelbox, "tanimoto_nn"):
            candidates_ds = modelbox.tanimoto_nn(
                data=candidates_ds,
                query_structure_column=args.structure,
                query_input_representation=args.input_representation,
                **common_args,
            )
            preprocessing_args["_extra_cols_to_keep"].append(modelbox.tanimoto_column)
        else:
            print_err(f"Cannot calculate Tanimoto for non-chemical modelbox from {args.checkpoint}")
    if args.doubtscore:
        modelbox.model.set_model(0)
        candidates_ds = modelbox.doubtscore(
            candidates=candidates_ds,
            features=features,
            context=context,
            preprocessing_args=preprocessing_args,
            last_layer_only=args.last_layer,
            **common_args,
        )
        preprocessing_args["_extra_cols_to_keep"].append("doubtscore")
    if args.information_sensitivity:
        modelbox.model.set_model(0)
        if args.approx == "bekas":
            extra_args = {"n": args.bekas_n}
        else:
            extra_args = {}
        candidates_ds = modelbox.information_sensitivity(
            candidates=candidates_ds,
            features=features,
            context=context,
            preprocessing_args=preprocessing_args,
            approximator=args.approx,
            optimality_approximation=args.optimality,
            last_layer_only=args.last_layer,
            **common_args,
            **extra_args,
        )
        preprocessing_args["_extra_cols_to_keep"].append("information sensitivity")
        
    print_err(preprocessing_args)

    _save_dataset(
        candidates_ds.remove_columns([
            col for col in candidates_ds.column_names 
            if col.startswith(modelbox._in_key)
        ] + [modelbox._out_key]), 
        output,
    )

    if args.labels is not None:
        overall_metrics = defaultdict(list)
        metric_filename = os.path.join(out_dir, "predict-eval-metrics-table.csv")
        plot_filename = os.path.join(out_dir, "predict-eval-scatter")
        metrics = _evaluate_modelbox_and_save_metrics(
            modelbox,
            dataset=candidates_ds,
            **preprocessing_args,
            metric_filename=metric_filename,
            plot_filename=plot_filename,
        )
        pprint_dict(
            metrics, 
            message=f"Evaluation",
        )
        overall_metrics["model_class"].append(modelbox.class_name)
        overall_metrics["n_parameters"].append(modelbox.size)
        keys_added = set(overall_metrics.keys())
        for d in (
            modelbox._init_kwargs, 
            modelbox._model_config, 
            metrics,
        ):
            for key, val in d.items():
                if key != "trainer_opts" and key not in keys_added:
                    overall_metrics[key].append(val)
                    keys_added.add(key)
        _dict_to_pandas(
            overall_metrics, 
            os.path.join(out_dir, "metrics.csv"),
        )

    return None