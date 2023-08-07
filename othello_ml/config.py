import os
import pathlib
import toml
import copy

# need something fancier for styles of model, maybe just a constructor and array of args?

model_styles = {
    "cnn_small": {
        "model_name": "CNN",
        "model_params": {"N1": 64, "N2": 64},
    },
    "cnn_medium": {
        "model_name": "CNN",
        "model_params": {"N1": 128, "N2": 64},
    },
    "cnn_large": {
        "model_name": "CNN",
        "model_params": {"N1": 1024, "N2": 64},
    },
    "small": {
        "model_name": "NN",
        "model_params": {"N1": 1024, "N2": 1024},
    },
    "small_norm": {
        "model_name": "NN_with_norm",
        "model_params": {"N1": 1024, "N2": 1024},
    },
    "medium": {
        "model_name": "NN",
        "model_params": {"N1": 2048, "N2": 1024},
    },
    "medium_norm": {
        "model_name": "NN_with_norm",
        "model_params": {"N1": 2048, "N2": 1024},
    },
    "large": {
        "model_name": "NN",
        "model_params": {"N1": 2048, "N2": 2048},
    },
    "large_norm": {
        "model_name": "NN_with_norm",
        "model_params": {"N1": 2048, "N2": 2048},
    },
}

def make_config(
        experiment_root, experiment_name, wthor_dir="/var/nfs/dpzmick/othello/wthor_files",
        debug=True,
        # data gen args
        board_lookback=0, include_flips=True,
        # split args
        train_perc=0.8,
        # train args
        model_style="small", loss_variant="loss_without_invalid", train_epochs=128,
        weight_decay=0.0, batch_size=512):

    name = f'{experiment_name}_{model_style}'
    if debug:
        name += '_debug'
    else:
        name += '_full'

    # salt the name with a timestamp or something?

    experiment_dir = f'{experiment_root}/{name}/'

    wthor_filenames = []
    for f in os.listdir(wthor_dir):
        if f.endswith(".wtb"):
            wthor_filenames.append(f)

    # order matters here, sort alpha numerically
    # we must keep the order the same across experiments
    wthor_filenames = list(map(lambda e: wthor_dir + '/' + e, sorted(wthor_filenames)))

    if debug:
        wthor_filenames = wthor_filenames[0:10]

    config = {
        "name": name,
        "experiment_dir": experiment_dir,
        "log_dir": experiment_dir + '/logs',
        "files": {
            # data gen inputs
            "wthor_files": wthor_filenames,
            # data gen outputs / test/train split inputs
            "ids_filename": f'{experiment_dir}/datasets/ids.dat.zst',
            "boards_dir": f'{experiment_dir}/datasets/boards/',
            "policy_filename": f'{experiment_dir}/datasets/policy.dat.zst',
            # test/train split outputs / training inputs
            "split_filename": f'{experiment_dir}/datasets/split.pt.lz4',
            "trace_file": f'{experiment_dir}/trace.trace',
            "checkpoint_dir": f'{experiment_dir}/checkpoints/'
        },
        "settings": {
            "boards_per_file": 500_000,
            "board_lookback": board_lookback, # how many previous boards to include. 0 means "only current board"
            "include_flips": include_flips,
            "train_perc": train_perc, # test/train split
            "batch_size": batch_size, # training
            "loss_variant": loss_variant,
            "weight_decay": weight_decay,
            "train_epochs": 16 if debug else train_epochs,
        }
    }

    # have to copy to avoid accidentally updating the template when modifying
    # the array to include nn input shape
    config["settings"].update(copy.deepcopy(model_styles[model_style]))
    config["settings"]["model_params"]["input_shape"] = 1+64+128+(128*board_lookback)

    return config

def setup(config):
    pathlib.Path(config["experiment_dir"]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config["experiment_dir"] + '/datasets').mkdir(parents=True, exist_ok=True)
    pathlib.Path(config["experiment_dir"] + '/checkpoints').mkdir(parents=True, exist_ok=True)
    pathlib.Path(config["log_dir"]).mkdir(parents=True, exist_ok=True)

    with open(config["experiment_dir"] + '/config.toml', "w") as f:
        toml.dump(config, f)

def open_config(experiment_dir):
    with open(experiment_dir + '/config.toml', "r") as f:
        return toml.load(f)
