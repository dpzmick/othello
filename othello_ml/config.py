import os
import pathlib
import pickle
import toml

class WThorConfig(object):
    # FIXME add some simplified setting for a debug config
    def __init__(self, data_dir):
        filenames = []

        for f in os.listdir(data_dir):
            if f.endswith(".wtb"):
                filenames.append(f)

        # order matters here, sort alpha numerically
        # we must keep the order the same across experiments
        self.filenames = list(map(lambda e: data_dir + '/' + e, sorted(filenames)))
        self.filenames = self.filenames[0:10]


class DatasetConfig(object):
    def __init__(self, include_flips):
        self.include_flips = include_flips

    def ids_filename(self, datadir):
        return f"{datadir}/ids.dat"

    def boards_filename(self, datadir):
        return f"{datadir}/boards.dat"

    def policy_filename(self, datadir):
        return f"{datadir}/policy.dat"

    def data_gen_toml_filename(self, datadir):
        return f"{datadir}/config.toml"

    def make_data_gen_toml(self, datadir, wthor_config):
        config = {
            "inputs": {
                "filenames": wthor_config.filenames
            },
            "outputs": {
                "ids_filename":    self.ids_filename(datadir),
                "boards_filename": self.boards_filename(datadir),
                "policy_filename": self.policy_filename(datadir),
            },
            "settings": {
                "include_flips": self.include_flips,
            }
        }

        with open(self.data_gen_toml_filename(datadir), "w") as f:
            toml.dump(config, f)

    def make_data_gen_command(self, datadir):
        return f"/var/nfs/dpzmick/othello/build/apps/data_gen_policy {self.data_gen_toml_filename(datadir)}"

    def make_compress_commands(self, datadir):
        for filename in [self.ids_filename(datadir), self.boards_filename(datadir), self.policy_filename(datadir)]:
            yield f"zstd -f {filename}"


class NetConfig(object):
    def __init__(self, batch_size, N1, N2):
        self.batch_size = batch_size
        self.N1 = N1
        self.N2 = N2


class ExperimentConfig(object):
    def __init__(self, name, experiment_root, train_perc, wthor_config, dataset_config, net_config):
        self.name = name
        self.experiment_dir = experiment_root + "/" + name
        self.train_perc = train_perc

        self.wthor_config = wthor_config
        self.dataset_config = dataset_config
        self.net_config = net_config

    @classmethod
    def from_experiment_dir(cls, experiment_dir):
        with open(f"{experiment_dir}/config.pkl", "rb") as f:
            return pickle.load(f)

    def datadir(self):
        return self.experiment_dir + '/datasets'

    def compressed_ids_filename(self):
        return self.dataset_config.ids_filename(self.datadir()) + '.zst'

    def compressed_boards_filename(self):
        return self.dataset_config.boards_filename(self.datadir()) + '.zst'

    def compressed_policy_filename(self):
        return self.dataset_config.policy_filename(self.datadir()) + '.zst'

    def compressed_split_filename(self):
        return f'{self.datadir()}/split.pt.lz4'

    def setup(self):
        pathlib.Path(self.datadir()).mkdir(parents=True, exist_ok=True)

        with open(f"{self.experiment_dir}/config.pkl", "wb") as f:
            pickle.dump(self, f)

        self.dataset_config.make_data_gen_toml(self.datadir(), self.wthor_config)

    def data_gen_command(self):
        return self.dataset_config.make_data_gen_command(self.datadir())

    def compression_commands(self):
        return self.dataset_config.make_compress_commands(self.datadir())
