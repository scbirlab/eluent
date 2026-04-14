from typing import Optional

from ..checkpoint_utils import save_json


def _evaluate_modelbox_and_save_metrics(
    modelbox,
    metric_filename: str,
    plot_filename: str,
    dataset: Optional = None,
    **kwargs
):
    import torch
    from ..utils.plotting import _plot_prediction_scatter

    modelbox.to("cuda" if torch.cuda.is_available() else "cpu")
    predictions, metrics = modelbox.evaluate(
        data=dataset,
        aggregator="mean",
        agg_kwargs={"keepdims": True},
        **kwargs,
    )
    save_json(
        metrics, 
        metric_filename,
    )
    _plot_prediction_scatter(
        predictions,
        x=modelbox._prediction_key,
        y=modelbox._out_key,
        filename=plot_filename,
    )
    return metrics | {
        "model_class": modelbox.class_name,
        "n_parameters": modelbox.size,
    }
