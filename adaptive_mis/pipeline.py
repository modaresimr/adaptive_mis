from __future__ import print_function, division
from sklearn.metrics import ConfusionMatrixDisplay

from IPython.display import display
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
        project_name=config.get('dataset', {}).get('title') or 'general',
        workspace="modaresimr",
        log_code=True,
        log_graph=True,
        disabled=not config['run'].get('comet', False),
        auto_param_logging=True,  # Can be True or False
        auto_histogram_tensorboard_logging=True,  # Can be True or False
        auto_metric_logging=True  # Can be True or False
    )
    experiment.add_tag(key)
    experiment.add_tag(config.get('model', {}).get('title') or 'None')
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
        experiment.log_image(img[0], image_channels="first", name=f"{mode}/img",)
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
            torchmetrics.ConfusionMatrix(num_classes=num_classes),
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
    os.makedirs(config['run']['save_dir'], exist_ok=True)
    save_config(config, config['run']['save_dir'] + "/config.yaml")

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
    final_res = []
    cms = []
    for tr_dataloader, vl_dataloader, te_dataloader, fold in evaluation.next():
        print(f"~~~~~~~~~~~~~~~ Fold {fold}/{evaluation.count()-1} ~~~~~~~~~~~~~~~~~")
        save_dir = f"{config['run']['save_dir']}/k{evaluation.count()-1}_{fold}"
        os.makedirs(save_dir, exist_ok=True)
        # g = torch.Generator()
        # g.manual_seed(0)
        # from .common.utils import seed_worker
        # # prepare train dataloader
        # tr_dataloader = DataLoader(tr_dataset, worker_init_fn=seed_worker, generator=g, **config['data_loader']['train'])
        # vl_dataloader = DataLoader(vl_dataset, worker_init_fn=seed_worker, generator=g, **config['data_loader']['validation'])
        # te_dataloader = DataLoader(te_dataset, worker_init_fn=seed_worker, generator=g, **config['data_loader']['test'])

        with experiment.train():
            log_sample_image(tr_dataloader, "train")
        with experiment.validate():
            log_sample_image(vl_dataloader, "validation")
        with experiment.test():
            log_sample_image(te_dataloader, "test")

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

        model_path = f"{save_dir}/model_state_dict.pt"

        if config['model']['load_weights']:
            model.load_state_dict(torch.load(model_path))
            print("Loaded pre-trained weights...")

        tr_prms = config['training']
        optimizer = globals()[tr_prms['optimizer']['name']](model.parameters(), **tr_prms['optimizer']['params'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **tr_prms['scheduler'])
        # optimizer=loader(config,"optimzier")
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
                num_classes=num_classes,
                fold=fold
            )
            log_model(experiment, best_model, model_name=f"Model_f{fold}")

        with experiment.test():
            te_metrics = test(best_model, te_dataloader, num_classes, config)
            metrics = serialize_metrics(te_metrics.compute())
            cm = metrics.pop('test_ConfusionMatrix')
            cms.append(cm)
            experiment.log_metrics(metrics, prefix=f"{fold}")
            experiment.log_confusion_matrix(matrix=cm, file_name=f"test_{fold}", title=f"test_{fold}", labels=dataset.class_names)

            # print("TEST====", metrics)
            metrics_dic = {k.replace("test_", ""): v for k, v in metrics.items()}
            metrics_dic = {'fold': fold, **metrics_dic}
            final_res.append(metrics_dic)
            df = pd.DataFrame([metrics_dic]).set_index("fold")
            display(df)
            with open(f"{save_dir}/test.json", 'w') as f:
                json.dump(metrics_dic, f, indent=4)

            experiment.end()
        # df = pd.DataFrame({k.replace("test_metrics/", ""): v for k, v in metrics.items()}, index=[0])

        # display(df)
            experiment.log_table(f"result{fold}.json", tabular_data=df, headers=True)

    avg_cm = np.average(cms, axis=0)
    df_cm = pd.DataFrame(cm, index=dataset.class_names, columns=dataset.class_names)
    print(df_cm)
    ConfusionMatrixDisplay(avg_cm, display_labels=dataset.class_names).plot()
    plt.savefig(save_dir + '/confusion_matrix.png')
    experiment.log_confusion_matrix(matrix=avg_cm, title=f"avg_{fold}", labels=dataset.class_names)

    df = pd.DataFrame(final_res).set_index("fold")
    df.loc['avg'] = df.mean()
    df.to_csv(f"{config['run']['save_dir']}/full_test.csv")
    display(df)
    experiment.log_table(f"result_full.json", tabular_data=df, headers=True)
    with open(config['run']['save_dir'] + "/completed", 'w') as f:
        f.write("1")


def serialize_metrics(computed_metrics):
    res = {}
    for k, v in computed_metrics.items():
        v = v.cpu().detach().numpy()
        try:
            res[k] = float(v)
        except:
            res[k] = v
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
        tq = tqdm(vl_dataloader, disable=True)
        tq.set_description('Checking Val...')
        for batch, batch_data in enumerate(tq):
            cnt += batch_data['mask'].shape[0]

            loss, preds_, msks_ = process_batch(batch_data, model, device, num_classes, config, criterion)
            losses.append(loss.item())
            evaluator.update(preds_, msks_)
        tq.close()
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
        num_classes,
        fold,
        save_dir,
        save_file_id

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
    early_stopper = EarlyStopping(monitor='vl_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)
    fieldnames = ['epoch', 'tr_loss', 'vl_loss']  # Add more field names based on your metrics
    csv_logger = CSVLogger(f'{save_dir}/training_log.csv', fieldnames=fieldnames, separator=';', append=True)
    tqdmcallback = TqdmCallback(verbose=1)
    tqdmcallback.set_params({'epochs': epochs, 'steps': len(tr_dataloader)})

    for epoch in range(epochs):
        tqdmcallback.on_epoch_begin(epoch)

        model.train()
        train_metrics.reset()

        # tr_iterator = tqdm()
        tr_losses = []
        cnt = 0

        for batch, batch_data in enumerate(tr_dataloader):
            imgs = batch_data['image'].to(device)
            cnt += imgs.shape[0]

            optimizer.zero_grad()
            loss, preds_, msks_ = process_batch(batch_data, model, device, num_classes, config, criterion)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

            train_metrics.update(preds_, msks_)
            descr = f"Train) ep:{epoch:03d}, batch:{batch + 1:04d} -> curr_ml:{np.sum(tr_losses) / cnt:0.5f}, mbatch_l:{tr_losses[-1] / imgs.shape[0]:0.5f}"
            # tr_iterator.set_description(descr)
            tqdmcallback.on_batch_end(batch, descr)

        tr_loss = np.sum(tr_losses) / cnt
        tr_metrics = serialize_metrics(train_metrics.compute())
        tr_cm = tr_metrics.pop('train_ConfusionMatrix')

        with experiment.validate():
            vl_metrics, vl_loss = validate(model, criterion, vl_dataloader, valid_metrics, num_classes, config)
            vl_metrics = serialize_metrics(vl_metrics.compute())
            vl_cm = vl_metrics.pop('valid_ConfusionMatrix')
            experiment.log_confusion_matrix(matrix=vl_cm, epoch=epoch, step=epoch, title=f"val_{fold}", file_name=f"val_{fold}")

        epoch_info = {
            'tr_loss': tr_loss,
            'vl_loss': vl_loss,
            'tr_metrics': tr_metrics,
            'vl_metrics': vl_metrics,
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
        # train_metrics.reset()
        scheduler.step(vl_loss)
        info = f"trl={tr_loss:0.5f} vll={vl_loss:0.5f} best_vll={best_vl_loss:0.5f}"
        tqdmcallback.on_epoch_end(epoch, info)
        experiment.log_confusion_matrix(matrix=tr_cm, epoch=epoch, step=epoch, title=f"train_{fold}", file_name=f"train_{fold}")

    # Save results and models
    res = {
        'id': save_file_id,
        'config': config,
        'epochs_info': epochs_info,
        'best_result': best_result
    }
    experiment.log_curve("train_loss_{fold}", [i for i, e in enumerate(epochs_info)], [e['tr_loss'] for e in epochs_info])
    experiment.log_curve("val_loss_{fold}", [i for i, e in enumerate(epochs_info)], [e['vl_loss'] for e in epochs_info])
    with open(os.path.join(save_dir, f"{save_file_id + '_' if save_file_id else ''}result.json"), "w") as write_file:
        json.dump(res, write_file, indent=4)

    torch.save(model.state_dict(), os.path.join(save_dir, "last_model_state_dict.pt"))
    torch.save(best_model.state_dict(), os.path.join(save_dir, "best_model_state_dict.pt"))

    return best_model, model, res


def test(model, te_dataloader, num_classes, config):
    model.eval()
    test_metrics = get_metrics(num_classes).clone(prefix='test_').to(device)
    with torch.no_grad():
        evaluator = test_metrics.clone().to(device)
        tq = tqdm(te_dataloader)
        tq.set_description('Testing...')
        for batch_data in tq:
            _, preds_, msks_ = process_batch(batch_data, model, device, num_classes, config)
            evaluator.update(preds_, msks_)

    return evaluator
