import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, metrics


def inference_loop(net, dataset: datasets.AbstractSpaceNet7Dataset, device: torch.device,
                   multimodal: bool = False) -> dict:

    net.to(device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer = metrics.MultiThresholdMetric(thresholds)

    data = {'change': [], 'semantics': []}

    dataloader = torch_data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)
    with torch.no_grad():
        for step, item in enumerate(dataloader):

            if multimodal:
                x_m1 = item['x_m1'].to(device)
                x_m2 = item['x_m2'].to(device)
                logits, _ = net(x_m1, x_m2)
            else:
                x = item['x'].to(device)
                logits = net(x)
            y_hat = torch.sigmoid(logits)

            y = item['y'].to(device)
            measurer.add_sample(y.detach(), y_hat.detach())

    f1s = measurer.compute_f1()
    precisions, recalls = measurer.precision, measurer.recall
    data['change'].append(f1s.max().item())
    argmax_f1 = f1s.argmax()
    data['change'].append(precisions[argmax_f1].item())
    data['change'].append(recalls[argmax_f1].item())

    return data


def model_evaluation(net, cfg, device: torch.device, run_type: str, epoch: float, step: int,
                     multimodal: bool = False) -> float:

    ds = datasets.SpaceNet7EvaluationDataset(cfg, run_type)

    data = inference_loop(net, ds, device, multimodal=multimodal)

    f1_change, precision_change, recall_change = data['change']
    wandb.log({f'{run_type} change F1': f1_change,
               f'{run_type} change precision': precision_change,
           f'{run_type} change recall': recall_change,
               'step': step, 'epoch': epoch,
               })

    return f1_change


def model_evaluation_earlystopping(net, cfg, device: torch.device, run_type: str, multimodal: bool = False):
    ds = datasets.SpaceNet7EvaluationDataset(cfg, run_type)

    data = inference_loop(net, ds, device, multimodal)

    f1_change, precision_change, recall_change = data['change']
    wandb.log({f'earlystopping {run_type} change F1': f1_change,
               f'earlystopping {run_type} change precision': precision_change,
               f'earlystopping {run_type} change recall': recall_change,
               })
