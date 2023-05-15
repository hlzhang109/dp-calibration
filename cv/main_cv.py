"""CIFAR-10 classification with Vi-T."""
import logging
from unittest import result

import fire
import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers
from swissknife import utils
from torchvision import transforms
import numpy as np
from os.path import join
from private_transformers.transformers_support import freeze_isolated_params_for_vit
from private_transformers.privacy_utils.privacy_engine import PrivacyEngine
from datasets import load_dataset
import argparse
import os 
from torchvision import models
import torch.nn as nn

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

@torch.no_grad()
def evaluate(loader, model, args):
    model.eval()
    xents, accs = [], []
    for i, (images, labels) in enumerate(loader):
        images, labels = tuple(t.to(device) for t in (images, labels))
        #logits = model(pixel_values=images).logits
        if 'resnet' in args.model_name_or_path:
            logits = model(images) #.logits
        else:
            logits = model(pixel_values=images).logits
        y_pred = logits.argmax(dim=-1)
        xents.append(F.cross_entropy(logits, labels, reduction='none'))
        accs.append(y_pred.eq(labels).float())
    return tuple(torch.cat(lst).mean().item() for lst in (xents, accs))

def main(
    args,
    data_name='FGVCAircraft',
    train_batch_size=1000,
    per_device_train_batch_size=50,
    test_batch_size=100,
    epochs=10,
    lr=2e-3,
    max_grad_norm=0.1,
    linear_probe=True,
    target_acc=0,
    objective=100,
):
    model_name_or_path=args.model_name_or_path
    target_epsilon = args.target_epsilon
    gradient_accumulation_steps = train_batch_size // per_device_train_batch_size

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_loader, valid_loader, test_loader = utils.get_loader(
        data_name=data_name,
        task="classification",
        root=args.data_root,
        valid_ratio=0.1,  # validation set ratio * train set size = validation set size
        train_batch_size=per_device_train_batch_size,
        test_batch_size=test_batch_size,
        data_aug=False,
        drop_last=False,
        train_transform=image_transform,
        test_transform=image_transform,
    ) # 50_000, 10_000

    if 'resnet' in args.model_name_or_path: 
        model = models.resnet50(pretrained=True).to(device)
        model.fc = nn.Linear(2048, args.num_labels).to(device)
    else:
        config = transformers.AutoConfig.from_pretrained(model_name_or_path)
        config.num_labels = args.num_labels 
        model = transformers.ViTForImageClassification.from_pretrained(
            model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True  # Default pre-trained model has 1k classes; we only have 10.
        ).to(device)
    if linear_probe:
        model.requires_grad_(False)
        if 'resnet' in args.model_name_or_path:
            model.fc.requires_grad_(True)
        else:
            model.classifier.requires_grad_(True)
        logging.warning("Linear probe classification head.")
    else:
        freeze_isolated_params_for_vit(model) # private_transformers.
        logging.warning("Full fine-tune up to isolated embedding parameters.")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=args.wd_rate) # AdamW
    if args.no_private:
        logging.warning("No private training.")
    else:
        privacy_engine = PrivacyEngine( 
            model,
            batch_size=train_batch_size,
            sample_size=len(train_loader.dataset), 
            epochs=epochs,
            max_grad_norm=max_grad_norm,
            target_epsilon=target_epsilon,
        )
        privacy_engine.attach(optimizer)
        logging.warning("Private training with target epsilon: %s", target_epsilon)

    train_loss_meter = utils.AvgMeter()
    tepochs = tqdm(range(epochs), desc="Epochs")
    saved_model_name = "linear_probe" if linear_probe else "full_finetune"
    res_folder = 'nondp' if args.no_private else f'eps{int(target_epsilon)}'
    if args.same:
        res_folder = 'same'
    result_path = join('cv', saved_model_name, res_folder, data_name)
    if not os.path.exists(result_path):
        logging.warning(f"output_dir doesn't exists, mkdir now: {result_path}")
        os.makedirs(result_path)
    
    best_acc = 0
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
                            optimizer, num_warmup_steps=0, num_training_steps = epochs * len(train_loader.dataset) // train_batch_size,
                )
    for epoch in tepochs:
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader))
        for global_step, (images, labels) in pbar:
            model.train()
            images, labels = tuple(t.to(device) for t in (images, labels))
            if 'resnet' in args.model_name_or_path:
                logits = model(images) #.logits
            else:
                logits = model(pixel_values=images).logits
            loss = F.cross_entropy(logits, labels, reduction="none")
            train_loss_meter.step(loss.mean().item())
            if global_step % gradient_accumulation_steps == 0:
                if args.no_private:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                else:
                    optimizer.step(loss=loss)
                optimizer.zero_grad()
            else:
                if args.no_private:
                    pass
                else:
                    optimizer.virtual_step(loss=loss)
            #lr_scheduler.step()
            pbar.set_description(f"Train loss running average: {train_loss_meter.item():.4f}")
        # NOTE for ablation, use test_loader instead of valid_loader 
        if args.same:
            avg_xent, avg_acc = evaluate(test_loader, model, args)
            abs_value = abs(avg_acc - target_acc)
            if abs_value < objective:
                torch.save(model.state_dict(), join(result_path, f"{saved_model_name}.pth"))
                logging.warning(f"Best matched {target_acc} model accuracy {avg_acc} found at epoch {epoch}. Saved model to {join(result_path, f'{saved_model_name}.pth')}")
                objective = abs_value
        else:
            avg_xent, avg_acc = evaluate(valid_loader, model, args)
            if avg_acc > best_acc:
                torch.save(model.state_dict(), join(result_path, f"{saved_model_name}.pth"))
                logging.warning(f"Best model accuracy {avg_acc} found at epoch {epoch}. Saved model to {join(result_path, f'{saved_model_name}.pth')}")
                best_acc = avg_acc
        logging.warning(
            f"Epoch: {epoch}, lr: {get_lr(optimizer)}, average cross ent loss: {avg_xent:.4f}, average accuracy: {avg_acc:.4f}"
        )
    torch.save(train_loader, join(result_path, 'train_loader.pth'))
    torch.save(valid_loader, join(result_path,'valid_loader.pth'))
    torch.save(test_loader, join(result_path,'test_loader.pth'))

def inference(
    args,
    model_name_or_path='google/vit-base-patch16-224',
    data_name='FGVCAircraft',
    linear_probe=True,
):
    model_name_or_path = args.model_name_or_path
    target_epsilon = args.target_epsilon
    saved_model_name = "linear_probe" if linear_probe else "full_finetune"
    res_folder = 'nondp' if args.no_private else f'eps{int(target_epsilon)}'
    if args.same:
        res_folder = 'same'
    result_path = join('cv', saved_model_name, res_folder, data_name)
    print("Saved to ", result_path)
    valid_loader = torch.load(join(result_path,'valid_loader.pth'))
    test_loader = torch.load(join(result_path,'test_loader.pth'))

    if 'resnet' in model_name_or_path:
        model = models.resnet50(pretrained=True).to(device)
        model.fc = nn.Linear(2048, args.num_labels).to(device)
    else:
        config = transformers.AutoConfig.from_pretrained(model_name_or_path)
        config.num_labels = args.num_labels 
        model = transformers.ViTForImageClassification.from_pretrained(
            model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True  
        ).to(device)
    model.load_state_dict(torch.load(join(result_path, f"{saved_model_name}.pth")))
    model.eval()

    train_loss_meter = utils.AvgMeter()

    def eval_save(loader, note=None):
        logit_ls = []
        label_ls = []
        with torch.no_grad():
            pbar = tqdm(enumerate(loader, 1), total=len(loader))
            for global_step, (images, labels) in pbar:
                model.train()
                images, labels = tuple(t.to(device) for t in (images, labels))
                if 'resnet' in args.model_name_or_path:
                    logits = model(images) #.logits
                else:
                    logits = model(pixel_values=images).logits
                label_ls.append(labels.cpu().numpy())
                loss = F.cross_entropy(logits, labels, reduction="none")
                logit_ls.append(logits.cpu().numpy())
            avg_xent, avg_acc = evaluate(test_loader, model, args)
            logging.warning(
                f"Average cross ent loss: {avg_xent:.4f}, average accuracy: {avg_acc:.4f}"
            )
        logit_ls = np.concatenate(logit_ls, axis=0)
        np.save(join(result_path, f"{note}_logits.npy"), logit_ls) 
        logging.warning("Saved to {}".format(join(result_path, f"{note}_logits.npy")))
        label_ls = np.concatenate(label_ls, axis=0)
        np.save(join(result_path, f"{note}_labels.npy"), label_ls)
        logging.warning("Saved to {}".format(join(result_path, f"{note}_labels.npy")))

    eval_save(valid_loader, note="valid")
    eval_save(test_loader, note="test")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default='google/vit-base-patch16-224', type=str)
    parser.add_argument("--data_name", default="FGVCAircraft", type=str)
    parser.add_argument("--data_root", default="data root for downloading", type=str)
    parser.add_argument("--linear_probe", action="store_true")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--wd_rate", default=0, type=float) # 1e-4
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float) # 1e3
    parser.add_argument("--target_epsilon", default=3.0, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_labels", default=-1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--lr_decay", default=False, type=bool)
    parser.add_argument("--no_private", default=False, type=bool)
    parser.add_argument("--same", default=False, type=bool)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    set_seed(123)
    args = arg_parse()
    data_sizes = {'FGVCAircraft': 6001+666+3333, 'SUN397': 78_305+9787+9787, 'DTD': 1880+1880+1880, 
                  'StanfordCars': 7330+814+8041, 'Food101': 68_175+7575+25_250,  'cifar10': 50_000+10_000}
    data_epochs = {'FGVCAircraft': 100*10, 'SUN397': 40, 'DTD': 100*10, 'StanfordCars': 100*10, 'Food101': 20 , 'cifar10': 20} # 'Food101': 15
    num_labels = {'FGVCAircraft': 100, 'SUN397': 397,  'DTD': 47, 'StanfordCars': 196, 'Food101': 101, 'cifar10': 10}

    dp_lrs = {'FGVCAircraft': 1e-2, 'SUN397': 1e-2, 'DTD': 1e-2, 'StanfordCars': 1e-2, 'Food101': 1e-4, 'cifar10': 2e-3}
    nondp_lrs = {'FGVCAircraft': 1e-2, 'SUN397': 5e-1, 'DTD': 1e-2, 'StanfordCars': 1e-3, 'Food101': 5e-1, 'cifar10': 5e-1}

    batch_sizes = {'FGVCAircraft': 200, 'SUN397': 1000, 'DTD': 1000, 'StanfordCars': 200, 'Food101': 1000, 'cifar10': 1000}
    nondp_batch_sizes = {'FGVCAircraft': 200, 'SUN397': 128, 'DTD': 1000, 'StanfordCars': 200, 'Food101': 128, 'cifar10': 256}
    
    same_accs = {'FGVCAircraft': 0.9, 'SUN397': 0.6844, 'DTD': 0.9, 'StanfordCars': 0.9, 'Food101': 0.7582, 'cifar10': 0.7951}

    data = args.data_name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.no_private:
        print("using nondp lrs")
        lrs = nondp_lrs
        batch_sizes = nondp_batch_sizes
    else:
        lrs = dp_lrs
    args.lr = lrs[data]
    train_batch_size = batch_sizes[data]
    args.num_labels = num_labels[data]

    if data == 'cifar10':
        args.model_name_or_path = "microsoft/resnet-50"

    logging.warning(args)
    main(args, data_name=data, train_batch_size=train_batch_size, max_grad_norm=args.max_grad_norm, lr=args.lr, epochs=data_epochs[data], target_acc=same_accs[data], linear_probe=True)
    inference(args, data_name=data, linear_probe=True)