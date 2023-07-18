import os
import pathlib
import toml

# need something fancier for styles of model, maybe just a constructor and array of args?

model_styles = {
    "small": {
        "model_name": "NN",
        "model_params": {"N1": 1024, "N2": 1024},
    },
    "medium": {
        "model_name": "NN",
        "model_params": {"N1": 2048, "N2": 1024},
    },
    "large": {
        "model_name": "NN",
        "model_params": {"N1": 2048, "N2": 2048},
    },
}

def make_config(
        experiment_root, experiment_name, wthor_dir="/var/nfs/dpzmick/othello/wthor_files",
        debug=True,
        # data gen args
        include_flips=True,
        # split args
        train_perc=0.8,
        # train args
        model_style="small", loss_variant="loss_without_invalid", weight_decay=0.0, batch_size=2048):

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
            "boards_filename": f'{experiment_dir}/datasets/boards.dat.zst',
            "policy_filename": f'{experiment_dir}/datasets/policy.dat.zst',
            # test/train split outputs / training inputs
            "split_filename": f'{experiment_dir}/datasets/split.pt.lz4',
        },
        "settings": {
            "include_flips": include_flips,
            "train_perc": train_perc, # test/train split
            "batch_size": batch_size, # training
            "loss_variant": loss_variant,
            "weight_decay": weight_decay,
        }
    }

    config["settings"].update(model_styles[model_style])

    return config

def setup(config):
    pathlib.Path(config["experiment_dir"]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config["experiment_dir"] + '/datasets').mkdir(parents=True, exist_ok=True)
    pathlib.Path(config["log_dir"]).mkdir(parents=True, exist_ok=True)

    with open(config["experiment_dir"] + '/config.toml', "w") as f:
        toml.dump(config, f)

def open_config(experiment_dir):
    with open(experiment_dir + '/config.toml', "r") as f:
        return toml.load(f)
