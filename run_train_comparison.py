import json
import torch

from utils import DataModule, Trainer
from models import *


data_params = {
    "ea-mouse": dict(
        path='datasets/EA-Mouse/',
        num_chan=1, num_time=200, num_class=2,
        fractions=[1, 0.064, 0.016, 0.004, 0.001],
        sub_list=[f'{i:>2d}' for i in range(10, 30)],
        group_info={
            "G1": ['10', '15', '20', '25'],
            "G2": ['11', '16', '21', '26'],
            "G3": ['12', '17', '22', '27'],
            "G4": ['13', '18', '23', '28'],
            "G5": ['14', '19', '24', '29'],
        },
    ),
    "ea-human": dict(
        path='datasets/EA-Human/',
        num_chan=1, num_time=200, num_class=2,
        fractions=[1, 0.16, 0.04, 0.01],
        sub_list=[f'{i:>2d}' for i in range(10, 18)],
        group_info={f"G{s+1:d}": [f"1{s}"] for s in range(8)},
    ),
    "bonn": dict(
        path='datasets/Bonn/',
        num_chan=1, num_time=4719, num_class=2,
        fractions=[1],
        sub_list=[f'P{i:>1d}' for i in range(10)],
        group_info={f"G{s+1:0>2d}": [f"P{s}"] for s in range(10)},
    ),
    "baser": dict(
        path='datasets/Baser/',
        num_chan=2, num_time=1000, num_class=2,
        fractions=[1, 0.32, 0.08, 0.02],
        sub_list=['15', '16', '17', '86', '88',
                  '89', '90', '91', '92', '103', '104'],
        group_info={f"G{s:0>3s}": [s] for s in
                    ['15', '16', '17', '86', '88', '89', '90', '91', '92', '103', '104']},
    ),
    "iiic-seizure": dict(
        path='datasets/IIIC-Seizure/',
        num_chan=16, num_time=2000, num_class=6,
        fractions=[1, 0.16, 0.04],
        sub_list=[f'HQP{i:d}' for i in range(8)],
        group_info={f"G{s+1:d}": [f"HQP{s}"] for s in range(8)},
    ),
}

model_params = {
    "oceanet": dict(
        model=OCEANet,
        params=dict(
            patch_length=200, emb_type="residual", num_features=32,
            n_blocks=2, norm_group=8, kernel_size=7, encode_type='msata',
            emb_size=128, num_heads=8, depth=4, drop_rate=0.5,
        ),
    ),
    "eegnet": dict(
        model=EEGNet,
        params=dict(
            F1=8, F2=16, D=2, kernel_length=64, drop_rate=0.5, norm_rate=0.25,
        ),
    ),
    "sparcnet": dict(
        model=SPaRCNet,
        params=dict(
            n_blocks=7, out_channels_init=64, block_layers=4, growth_rate=16,
            conv_bias=True, batch_norm=False, bn_size=4, drop_rate=0.2,
        ),
    ),
    "eegconformer": dict(
        model=EEGConformer,
        params=dict(
            out_channels_init=40, emb_size=40, kernel_size=25,
            patch_length=75, hop_length=60, depth=6, num_heads=10,
        ),
    ),
    "eegdeformer": dict(
        model=EEGDeformer,
        params=dict(
            temporal_kernel=11, num_kernel=64, downsample=2,
            depth=4, heads=16, mlp_dim=16, dim_head=16,
        ),
    ),
    "biot": dict(
        model=BIOT,
        params=dict(
            emb_size=256, heads=8, depth=4, drop_rate=(0.2, 0.1),
        ),
    ),
    "cbramod": dict(
        model=CBraMod,
        params=dict(
            out_channels_init=25, emb_size=200, drop_rate=0.5, norm_group=5,
        ),
    ),
    #### ablation ####
    "oceanet-plain-cnn": dict(
        model=OCEANet,
        params=dict(
            patch_length=200, emb_type="shallow", num_features=32,
            n_blocks=2, norm_group=8, kernel_size=7, encode_type='msata',
            emb_size=128, num_heads=8, depth=4, drop_rate=0.5,
        ),
    ),
    "oceanet-ap": dict(
        model=OCEANet,
        params=dict(
            patch_length=200, emb_type="residual", num_features=32,
            n_blocks=2, norm_group=8, kernel_size=7, encode_type='none',
            emb_size=128, num_heads=8, depth=4, drop_rate=0.5,
        ),
    ),
    "oceanet-mlp": dict(
        model=OCEANet,
        params=dict(
            patch_length=200, emb_type="residual", num_features=32,
            n_blocks=2, norm_group=8, kernel_size=7, encode_type='mlp',
            emb_size=128, num_heads=8, depth=4, drop_rate=0.5,
        ),
    ),
    "oceanet-flat-msa": dict(
        model=OCEANet,
        params=dict(
            patch_length=200, emb_type="residual", num_features=32,
            n_blocks=2, norm_group=8, kernel_size=7, encode_type='msa',
            emb_size=128, num_heads=8, depth=4, drop_rate=0.5,
        ),
    ),
}

training_params = dict(
    batch_size=128,
    learning_rate=0.0005,
    patience=10,
)


if __name__ == '__main__':
    for dname, dparams in data_params.items():
        data_ = DataModule(data_path=dparams['path'],
                           sub_list=dparams['sub_list'],
                           shuffle=False)

        for fac in dparams['fractions']:
            data = DataModule(data_path=dparams['path'],
                              sub_list=dparams['sub_list'],
                              percent=fac)

            for mname, mparams in model_params.items():
                for gname, gsub in dparams['group_info'].items():
                    run_name = f"train_{dname}/{fac*1000:0>4d}/{mname}_{gname}"

                    with open(run_name + ".cfg", "w") as f:
                        configs = {"data_params": dparams,
                                   "model_params": mparams,
                                   "training_params": training_params,
                                   }
                        configs["model"] = mparams["model"].__name__
                        json.dump(configs, f, indent=4)

                    print(f"{run_name} training ---------->")

                    trainer = Trainer(
                        mparams['model'](in_channels=dparams['num_chan'],
                                         n_samples=dparams['num_time'],
                                         n_classes=dparams['num_class'],
                                         **mparams['params']),
                        **training_params,
                    )
                    trainer.load_training_data(data, gid=gsub, sid=[0])
                    trainer.train()

                    with open(run_name + ".log", "w") as f:
                        json.dump(trainer.training_log, f, indent=4)
                    torch.save(trainer.best_model, run_name + ".pth")

                    for sub in gsub:
                        print(f"{run_name}_{sub} test predicting ---------->")

                        trainer.load_test_data(data_, gid=[sub])
                        performance = trainer.evaluate()
                        with open(run_name + f"_{sub}.json", "w") as f:
                            json.dump(performance, f, indent=4)
