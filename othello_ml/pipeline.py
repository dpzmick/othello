from .config import make_config, setup
import numpy as np

# python3 -m othello_ml.pipeline | bash -

experiment_root="/var/nfs/dpzmick/experiments/"
debug = False

configs = [
    make_config(experiment_root, f"lookback_exp_{lookback}",
                debug=debug, include_flips=True,
                model_style=style,
                weight_decay=decay,
                board_lookback=lookback)

    for (style, decay) in [('small', 0), ('medium', 2e-4), ('large', 2e-4)]
    for lookback in [1, 3, 5]
]

# experiments to run:
# - get a loss function working that includes valid moves as part of the loss function
# - include some board history in the input (maybe requires even more giant of a model?)
# - add regularization

print(f"#!/bin/bash")
print(f"set -e")

# FIXME make sure all config names are unique

for c in configs:
    setup(c)

    name           = c['name']
    log_dir        = c['log_dir']
    experiment_dir = c['experiment_dir']
    config_path    = c['experiment_dir'] + '/config.toml' # FIXME change data gen to take just the dir

    print("echo -----------------------")
    print(f"echo Submitting jobs for {name}")

    # generate the raw dataset
    data_gen_cmd = f"./wrap.sh /var/nfs/dpzmick/othello/build/apps/data_gen_policy {config_path}"
    print(f"GEN=$(sbatch -J gen_{name} --parsable -N1 -n1 --mem-per-cpu=4G --output {log_dir}/gen.log {data_gen_cmd} )") # doesn't need wrap?
    print("echo gen job id: $GEN")

    # generate test train split, memory intensive
    split_cmd = f"./wrap.sh python -u -m othello_ml.scripts.make_split {experiment_dir}"
    print(f"SPLIT=$(sbatch -J split_{name} --parsable -N1 -n1 --output {log_dir}/split.log --dependency=afterok:$GEN {split_cmd} )")
    print(f"echo split job: $SPLIT")

    # run training
    train_cmd = f"./wrap.sh python -u -m othello_ml.scripts.train {experiment_dir}"
    print(f"TRAIN=$(sbatch -J train_{name} --parsable -N1 -n4 --mem-per-cpu=2G --output {log_dir}/train.log --dependency=afterok:$SPLIT {train_cmd} )")
    print(f"echo train job: $TRAIN")

# FIXME skip steps that are already done in this experiment dir
# FIXME the binary dir should probably also be configurable
