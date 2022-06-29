import os
import yaml
import itertools


def main():
    params = {
        "l_win": [50, 100],
        "batch_size": [64],
        "num_workers": [4],
        "n_head": [1],
        "dff": [64],
        "num_layers": [2],
        "lr": [0.0001],
        "weight_decay": [0.0001],
        "n_epochs": [1],
        "dropout": [0.2],
    }

    keys, values = zip(*params.items())
    combs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"GENERATING {len(combs)} NEW CONFIGS ...")

    for comb in combs:
        filename = "{}stacks_{}lwin_{}lr_{}batch_{}epcs".format(
            comb["num_layers"],
            comb["l_win"],
            comb["lr"],
            comb["batch_size"],
            comb["n_epochs"],
        ).replace(".", "_")
        config_path = os.path.join("configs/", "{}.yml".format(filename))
        config = {
            "experiment": filename,
            "l_win_max": 31,
            "d_model": 16,
        }
        config.update(comb)
        print(filename)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

    print("DONE.")


if __name__ == "__main__":
    main()
