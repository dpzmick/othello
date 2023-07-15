from .config import ExperimentConfig, WThorConfig, DatasetConfig, NetConfig

configs = [
    ExperimentConfig(
        name="with_flips",
        experiment_dir="/var/nfs/dpzmick/experiments/with_flips",
        train_perc=0.8,
        wthor_config=WThorConfig("/var/nfs/dpzmick/othello/wthor_files"),
        dataset_config=DatasetConfig(
            include_flips=True
        ),
        net_config=NetConfig(
            batch_size=2048,
            N1=2048, N2=1024,
        )
    ),

    ExperimentConfig(
        name="without_flips",
        experiment_dir="/var/nfs/dpzmick/experiments/without_flips",
        train_perc=0.8,
        wthor_config=WThorConfig("/var/nfs/dpzmick/othello/wthor_files"),
        dataset_config=DatasetConfig(
            include_flips=False
        ),
        net_config=NetConfig(
            batch_size=2048,
            N1=2048, N2=1024,
        )
    ),
]

print(f"#!/bin/bash")
print(f"set -e")

for c in configs:
    c.setup()

    print("echo -----------------------")
    print(f"echo Submitting jobs for {c.name}")

    # generate the raw dataset
    print(f"GEN=$(sbatch -J {c.name}_gen --parsable -N1 -n1 --output {c.datadir()}/gen.log ./wrap.sh {c.data_gen_command()} )");
    print("echo gen job id: $GEN")

    # compress everything
    for i, cmd in enumerate(c.compression_commands()):
        print(f"COMP_{i}=$(sbatch -J {c.name}_comp --dependency=afterok:$GEN --parsable -N1 -n1 --output {c.datadir()}/comp_{i}.log ./wrap.sh {cmd} )");
        print(f"echo compresion job {i} id $COMP_{i}")

    # generate test train split
    deps='afterok:' + ':'.join(f'$COMP_{i}' for i,_ in enumerate(c.compression_commands()))
    print(f"SPLIT=$(sbatch -J {c.name}_split --parsable -N1 -n1 --output {c.datadir()}/split.log --dependency={deps} ./wrap.sh python -m othello_ml.scripts.make_split {c.experiment_dir})")
    print(f"echo split job: $SPLIT")

    # run training
    print(f"TRAIN=$(sbatch -J {c.name}_train --parsable -N1 -n4 --output {c.experiment_dir}/train.log --dependency=afterok:$SPLIT ./wrap.sh python -m othello_ml.scripts.train {c.experiment_dir})")
    print(f"echo train job: $TRAIN")


# FIXME don't really like any of this very much unfortunately
# FIXME set memory limits on the jobs?
# FIXME the binary dir should probably also be configurable
# FIXME not really happy with this at all
# FIXME these are not really configs
