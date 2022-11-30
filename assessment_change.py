import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from utils import experiment_manager, datasets, evaluation
from networks import networks
from torch.utils import data as torch_data


def qualitative_assessment(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.TRAINER.EPOCHS, cfg, device)
    net.eval()
    dataset = datasets.SpaceNet7EvaluationDataset(cfg, run_type)
    dataloader = torch_data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    for i, item in enumerate(dataloader):
        x = item['x'].to(device)
        with torch.no_grad():
            logits = net(x)

        y_hat = torch.sigmoid(logits).detach().squeeze().cpu().numpy()
        y = item['y'].squeeze().cpu().numpy()

        x_start = x[:, 0, ].squeeze().cpu().numpy().transpose((1, 2, 0))
        x_end = x[:, -1, ].squeeze().cpu().numpy().transpose((1, 2, 0))

        fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
        axs[0].imshow(x_start)
        axs[1].imshow(x_end)

        axs[2].imshow(y, cmap='gray')
        axs[3].imshow(y_hat, cmap='gray')

        for _, ax in np.ndenumerate(axs):
            ax.set_axis_off()

        out_folder = Path(cfg.PATHS.OUTPUT) / 'plots' / cfg.NAME
        out_folder.mkdir(exist_ok=True)
        out_file = out_folder / f'{run_type}_{i}.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


def quantitative_assessment(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    print(cfg.NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)

    ds = datasets.SpaceNet7CDDataset(cfg, run_type, no_augmentations=True, dataset_mode='first_last',
                                     disable_multiplier=True, disable_unlabeled=True)

    data = evaluation.inference_loop(net, ds, device, False)
    f1, precision, recall = data['change']
    print(f'F1 score: {f1:.3f} - Precision: {precision:.3f} - Recall {recall:.3f}')


def assessment_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser(description="Experiment Args")

    parser.add_argument('-c', "--config-file", dest='config_file', required=True, help="path to config file")
    parser.add_argument('-o', "--output-dir", dest='output_dir', required=True, help="path to output directory")
    parser.add_argument('-d', "--dataset-dir", dest='dataset_dir', default="", required=True,
                        help="path to output directory")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = assessment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    qualitative_assessment(cfg)
