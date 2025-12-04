import json
import torch

from utils import DataModule, Trainer
from models import *

data_params = {
    "chb-mit": dict(
        path='datasets/CHB-MIT/',
        num_chan=18, num_time=2000, num_class=2,
        fractions=[1],
        sub_list=[f'chb{i:0>2d}' for i in range(1, 25)],
        group_info={f"G{s:0>2d}": [f"chb{s:0>2d}"] for s in range(1, 25)},
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
    "oceanet-pretrain-ea": dict(
        model=OCEANet,
        params=dict(
            patch_length=200, emb_type="residual", num_features=32,
            n_blocks=2, norm_group=8, kernel_size=7, encode_type='msata',
            emb_size=128, num_heads=8, depth=4, drop_rate=0.5,
            use_pretrained="oceanet_all_ea.pth",
        ),
    ),
    "oceanet-pretrain-iiic": dict(
        model=OCEANet,
        params=dict(
            patch_length=200, emb_type="residual", num_features=32,
            n_blocks=2, norm_group=8, kernel_size=7, encode_type='msata',
            emb_size=128, num_heads=8, depth=4, drop_rate=0.5,
            use_pretrained="oceanet_all_iiic.pth",
        ),
    ),
    "sparcnet": dict(
        model=SPaRCNet,
        params=dict(
            n_blocks=7, out_channels_init=64, block_layers=4, growth_rate=16,
            conv_bias=True, batch_norm=False, bn_size=4, drop_rate=0.2,
        ),
    ),
    "biot": dict(
        model=BIOT,
        params=dict(
            emb_size=256, heads=8, depth=4, drop_rate=(0.2, 0.1),
        ),
    ),
}

training_params = dict(
    batch_size=16,
    learning_rate=0.0005,
    patience=10,
)


if __name__ == '__main__':
    for dname, dparams in data_params.items():
        data_ = DataModule(data_path=dparams['path'] + 'test/',
                           sub_list=dparams['sub_list'],
                           shuffle=False)

        data = DataModule(data_path=dparams['path'] + 'training/',
                          sub_list=dparams['sub_list'])

        for mname, mparams in model_params.items():
            for gname, gsub in dparams['group_info'].items():
                run_name = f"train_{dname}/{mname}_{gname}"

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
