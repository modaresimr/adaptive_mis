from __future__ import print_function, division

from adaptive_mis.train.csv_logger import CSVLogger
from adaptive_mis.train.tqdm_callback import TqdmCallback
from .train import EarlyStopping
import csv
from copy import deepcopy
from PIL import Image
import cv2
import comet_ml
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from comet_ml import ConfusionMatrix, Experiment, init
from comet_ml.integration.pytorch import log_model
import datetime
import random
import string
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch.optim import Adam, SGD
from .loss.dice import DiceLoss, DiceLossWithLogtis
from torch.nn import BCELoss, CrossEntropyLoss
import os
import sys
import copy
import json
import importlib
import glob
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from tqdm import tqdm
from . import loader
from .common.utils import printc
from .common.config import save_config, load_config
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch device: {device}")


def setup_comet(config):
    global experiment
    key = config.get('run', {}).get('key') or 'None'
    experiment = Experiment(
        api_key="8v599AWHmFIfK0YPnIeHWUuwE",
        project_name=config.get('comet', {}).get('project_name') or 'general',
        workspace="modaresimr",
        log_code=True,
        log_graph=True,
        disabled=True,
        auto_param_logging=True,  # Can be True or False
        auto_histogram_tensorboard_logging=True,  # Can be True or False
        auto_metric_logging=True  # Can be True or False
    )
    experiment.add_tag(key)
    experiment.add_tag(config.get('model', {}).get('class') or 'None')
    experiment.log_parameters(config)
    return experiment


def set_seed():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)


def log_sample_image(dataloader, mode):
    for sample in dataloader:
        # print('sample', sample.shape)
        img = sample['image']
        msk = sample['mask']
        # show_sbs(img[0], msk[0, 1])
        experiment.log_image(img[0], image_channels="first", name=f"{mode}/img")
        experiment.log_image(msk[0], image_channels="first", name=f"{mode}/gt")
        break


def get_metrics(num_classes):

    # %%
    if num_classes > 2:
        params = {'num_classes': num_classes, 'mdmc_reduce': 'samplewise', 'average': 'macro'}
    else:
        params = {}
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.F1Score(**params),
            torchmetrics.Accuracy(**{**params, 'average': 'micro'}),
            torchmetrics.Dice(**params),
            torchmetrics.Precision(**params),
            torchmetrics.Specificity(**params),
            torchmetrics.Recall(**params),
            # IoU
            torchmetrics.JaccardIndex(**{**params, 'num_classes': num_classes})
        ],
        prefix='train_metrics/',

    )
    return metrics


def get_loss():
    # criterion_dice = DiceLoss()
    criterion_dice = DiceLossWithLogtis()
    # criterion_ce  = BCELoss()
    criterion_ce = CrossEntropyLoss()

    def criterion(preds, masks):
        c_dice = criterion_dice(preds, masks)
        c_ce = criterion_ce(preds, masks)
        return 0.5 * c_dice + 0.5 * c_ce
    return criterion


def execute(config):
    print(json.dumps(config, indent=2))
    print(20 * "~-", "\n")
    setup_comet(config)

    set_seed()

    # ------------------- params --------------------
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    img_transform = transforms.Compose([transforms.ToTensor()])
    # transform for mask
    msk_transform = transforms.Compose([transforms.ToTensor()])
    msk_transform = torch.tensor

    dataset = loader(config, 'dataset')
    num_classes = dataset.num_classes
    evaluation = loader(config, 'evaluation', dataset=dataset, cfg_dataloader=config['data_loader'], num_classes=num_classes)
    for tr_dataloader, vl_dataloader, te_dataloader, fold in evaluation.next():
        print(f"~~~~~~~~~~~~~~~ Fold {fold} ~~~~~~~~~~~~~~~~~")
        save_dir = config['run']['save_dir'] + "_{fold}"
        os.makedirs(save_dir, exist_ok=True)
        # g = torch.Generator()
        # g.manual_seed(0)
        # from .common.utils import seed_worker
        # # prepare train dataloader
        # tr_dataloader = DataLoader(tr_dataset, worker_init_fn=seed_worker, generator=g, **config['data_loader']['train'])
        # vl_dataloader = DataLoader(vl_dataset, worker_init_fn=seed_worker, generator=g, **config['data_loader']['validation'])
        # te_dataloader = DataLoader(te_dataset, worker_init_fn=seed_worker, generator=g, **config['data_loader']['test'])

        log_sample_image(tr_dataloader, "train")
        log_sample_image(vl_dataloader, "validation")
        log_sample_image(te_dataloader, "test")

        # %% [markdown]
        # ### Device

        # %%

        # %%
        # download weights

        # !wget "https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz"
        # !mkdir -p ../model/vit_checkpoint/imagenet21k
        # !mv R50+ViT-B_16.npz ../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz

        model = loader(config, 'model', in_channels=dataset.in_channels, out_channels=dataset.num_classes, img_size=dataset.image_size)
        torch.cuda.empty_cache()
        model = model.to(device)
        number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of parameters:", number_of_parameters)
        experiment.log_parameter("Number of parameters", number_of_parameters)
        experiment.set_model_graph(str(model))
        save_config(config, save_dir + "/config.yaml")
        model_path = f"{save_dir}/model_state_dict.pt"

        if config['model']['load_weights']:
            model.load_state_dict(torch.load(model_path))
            print("Loaded pre-trained weights...")

        tr_prms = config['training']
        optimizer = globals()[tr_prms['optimizer']['name']](model.parameters(), **tr_prms['optimizer']['params'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **tr_prms['scheduler'])
        # optimizer=loader(config,"optimzier")
        # %%
        with experiment.train():
            best_model, model, res = train(
                model,
                tr_dataloader,
                vl_dataloader,
                config,
                optimizer,
                scheduler,
                save_dir=save_dir,
                save_file_id=None,
                num_classes=num_classes
            )
            log_model(experiment, best_model, model_name="TheModel")

        # %%
        with experiment.test():
            te_metrics = test(best_model, te_dataloader, num_classes, config)
            metrics = te_metrics.compute()
            experiment.log_metrics(metrics)

            print("TEST====", metrics)
            metrics_dic = {k.replace("test_metrics/", ""): float(v.cpu()) for k, v in metrics.items()}
            df = pd.DataFrame(metrics_dic, index=[0])
            from IPython.display import display
            display(df)
            with open(f"{save_dir}/test_final_result.json", 'w') as f:
                json.dump(metrics_dic, f, indent=4)

            experiment.end()

            metrics

        # %%
        print(metrics)
        df = pd.DataFrame({k.replace("test_metrics/", ""): v for k, v in metrics.items()}, index=[0])
        from IPython.display import display
        display(df)
        experiment.log_table("result.json", tabular_data=df, headers=True)

    # %%


def make_serializeable_metrics(computed_metrics):
    res = {}
    for k, v in computed_metrics.items():
        res[k] = float(v.cpu().detach().numpy())
    return res


def process_batch(batch_data, model, device, num_classes, config, criterion=None):
    imgs = batch_data['image'].to(device)
    msks = batch_data['mask'].to(device)
    preds = model(imgs)

    if criterion:
        loss = criterion(preds, msks)
    else:
        loss = None

    preds_ = torch.argmax(preds, 1, keepdim=False)
    if num_classes <= 2:
        preds_ = preds_.float()

    msks_ = torch.argmax(msks, 1, keepdim=False)

    if "SegPC2021" in config['dataset']['class']:
        not_nucs = torch.where(imgs[:, -1, :, :] > 0, 0, 1)
        preds_ = preds_ * not_nucs
        msks_ = (torch.tensor(msks_) * not_nucs).int()

    return loss, preds_, msks_


def validate(model, criterion, vl_dataloader, valid_metrics, num_classes, config):
    model.eval()
    with torch.no_grad():
        evaluator = valid_metrics.clone().to(device)

        losses = []
        cnt = 0.

        for batch, batch_data in enumerate(vl_dataloader):
            cnt += batch_data['mask'].shape[0]

            loss, preds_, msks_ = process_batch(batch_data, model, device, num_classes, config, criterion)
            losses.append(loss.item())
            evaluator.update(preds_, msks_)

        loss = np.sum(losses) / cnt
        # metrics = evaluator.compute()

    return evaluator, loss


def train(
        model,
        tr_dataloader,
        vl_dataloader,
        config,
        optimizer,
        scheduler,
        save_dir='./',
        save_file_id=None,
        num_classes=2
):
    criterion = get_loss()

    epochs = config['training']['epochs']
    torch.cuda.empty_cache()
    model = model.to(device)
    train_metrics = get_metrics(num_classes).clone(prefix='train_').to(device)
    valid_metrics = get_metrics(num_classes).clone(prefix='valid_').to(device)

    epochs_info = []
    best_model = None
    best_result = {}
    best_vl_loss = np.Inf
    early_stopper = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)
    fieldnames = ['epoch', 'tr_loss', 'vl_loss']  # Add more field names based on your metrics
    csv_logger = CSVLogger(f'{save_dir}/training_log.csv', fieldnames=fieldnames, separator=';', append=True)
    tqdmcallback = TqdmCallback(verbose=1)
    tqdmcallback.set_params({'epochs': epochs, 'steps': len(tr_dataloader)})

    for epoch in range(epochs):
        tqdmcallback.on_epoch_begin(epoch)

        model.train()
        train_metrics.reset()

        tr_iterator = tqdm(enumerate(tr_dataloader))
        tr_losses = []
        cnt = 0

        for batch, batch_data in tr_iterator:
            imgs = batch_data['image'].to(device)
            cnt += imgs.shape[0]

            optimizer.zero_grad()
            loss, preds_, msks_ = process_batch(batch_data, model, device, num_classes, config, criterion)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

            train_metrics.update(preds_, msks_)
            tr_iterator.set_description(
                f"Training) ep:{epoch:03d}, batch:{batch + 1:04d} -> curr_ml:{np.sum(tr_losses) / cnt:0.5f}, mbatch_l:{tr_losses[-1] / imgs.shape[0]:0.5f}")
            tqdmcallback.on_batch_end(batch)

        tr_loss = np.sum(tr_losses) / cnt
        vl_metrics, vl_loss = validate(model, criterion, vl_dataloader, valid_metrics, num_classes, config, typ='val', epoch=epoch)

        epoch_info = {
            'tr_loss': tr_loss,
            'vl_loss': vl_loss,
            'tr_metrics': make_serializeable_metrics(train_metrics.compute()),  # The method 'make_serializeable_metrics' should be defined or replaced
            'vl_metrics': make_serializeable_metrics(vl_metrics.compute()),   # The method 'make_serializeable_metrics' should be defined or replaced
        }

        csv_logger.log({'epoch': epoch, 'tr_loss': tr_loss, 'vl_loss': vl_loss})

        early_stopper(epoch_info, model)
        if early_stopper.early_stop:
            print("Early stopping")
            break
        if vl_loss < best_vl_loss:
            best_model = model
            best_vl_loss = vl_loss
            best_result = epoch_info

        # 'experiment' should be defined or replaced
        epochs_info.append(epoch_info)
        train_metrics.reset()
        scheduler.step(vl_loss)
        tqdmcallback.on_epoch_end(epoch)

    # Save results and models
    res = {
        'id': save_file_id,
        'config': config,
        'epochs_info': epochs_info,
        'best_result': best_result
    }

    with open(os.path.join(config['model']['save_dir'], f"{save_file_id + '_' if save_file_id else ''}result.json"), "w") as write_file:
        json.dump(res, write_file, indent=4)

    torch.save(model.state_dict(), os.path.join(config['model']['save_dir'], "last_model_state_dict.pt"))
    torch.save(best_model.state_dict(), os.path.join(config['model']['save_dir'], "best_model_state_dict.pt"))

    return best_model, model, res


def test(model, te_dataloader, num_classes, config):
    model.eval()
    test_metrics = get_metrics(num_classes).clone(prefix='test_').to(device)
    with torch.no_grad():
        evaluator = test_metrics.clone().to(device)

        for batch_data in tqdm(te_dataloader):
            _, preds_, msks_ = process_batch(batch_data, model, device, num_classes, config)
            evaluator.update(preds_, msks_)

    return evaluator
