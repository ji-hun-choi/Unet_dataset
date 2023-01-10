import wandb
import torch.nn as nn

from pathlib import Path
from torch import optim
import torch.nn.functional as F

from utils.utils import *
from utils.dice_score import *
from model.Unet import UNet
from evaluate import evaluate




def train(CFG, loader_train_list, loader_val_list, ckpt_dir, device, model):
    info_print('Start Train_Set')
    epochs, batch_size, learning_rate, img_scale, amp, gradient_clipping, momentum, patience = \
        CFG['epochs'], CFG['batch_size'], CFG['learning_rate'], CFG['img_scale'], CFG['amp'], CFG['gradient_clipping'],\
        CFG["momentum"], CFG['patience']
    val_percent = 1 - CFG['train_split_scale']
    save_checkpoint = ckpt_dir

    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    model = model.to(memory_format=torch.channels_last)
    model = model.to(device=device)

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    info_print('Strat Train Point')

    train_loader = loader_train_list[0]
    n_train = loader_train_list[1]

    val_loader = loader_val_list[0]
    n_val = loader_val_list[1]

    epochs = CFG['epochs']
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['input'], batch['label']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.squeeze(1).float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.squeeze(1).float(), multiclass=False)
                    else:

                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not torch.isinf(value).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not torch.isinf(value.grad).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        info_print('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'input': wandb.Image(images[0].cpu()),
                                'label': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            save(ckpt_dir=ckpt_dir, model=model, optim=optimizer, epoch=epoch)


if __name__ == '__main__':
    CFG = {
        'width': 400,
        'height': 400,
        'learning_rate': 1e-5,
        'batch_size': 4,
        'momentum': 0.999,
        'img_scale': 0.5,
        'train_split_scale': 0.75,
        'epochs': 100,
        'weight_decay': 1e-8,
        'patience': 8,
        'n_channels': 3,
        'n_classes': 1,
        'amp': False,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'data_dir': 'I:\\personal\\fire\\Valid',
        'project_dir': 'D:\\Projects\\Unet_dataset',
        'gradient_clipping': 1.0
    }

    data_dir = CFG['data_dir']
    info_print(f'data_dir = {data_dir}')

    ckpt_dir = os.path.join(CFG['project_dir'], 'checkpoint')
    info_print(f'checkpoint = {ckpt_dir}')

    log_dir = os.path.join(CFG['project_dir'], 'log')
    info_print(f'log = {log_dir}')

    device = CFG['device']

    loader_train, loader_val = load_data_set(cfg=CFG)
    model = UNet(n_channels=CFG['n_channels'], n_classes=CFG['n_classes'], bilinear=False)

    train(CFG, loader_train, loader_val, ckpt_dir, device, model)



