
from typing import Mapping, Optional, Union

from argparse import Namespace
import os

from carabiner import pprint_dict
from carabiner.cliutils import clicommand

@clicommand(message='Generating hyperparameter screening configurations')
def _hyperprep(args: Namespace) -> None:

    from ..hyperparameters import HyperOpt

    configs = HyperOpt.from_file(args.input_file)

    for i, config in enumerate(configs):
        pprint_dict(config, message=f"Configuration {i}")

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    configs.write(
        args.output, 
        serialize=args.serialize,
    )
        
    return None
