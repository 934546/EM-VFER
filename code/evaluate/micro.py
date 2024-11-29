from dataloader.SMEAD_dataset_me import Load_FEV_Dataset
from models import models_vit
import sys
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# from models.Generate_Model import GenerateModel
from models.Generate_Model import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import datetime
# from dataloader.video_dataloader import train_data_loader, test_data_loader
# from dataloader.emer_dataloader import Load_EMER_Dataset

from sklearn.metrics import confusion_matrix
import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CASME')

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=2)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int)

    parser.add_argument('--exper-name', type=str,default='CASME3')
    parser.add_argument('--temporal-layers', type=int, default=2)
    parser.add_argument('--img-size', type=int, default=224)

    args = parser.parse_args()
    return args

def main(set, args):
    
    data_set = 1
    
    model = GenerateModel(args=args)
    
    train_loader, val_loader = Load_FEV_Dataset(v_train_path=r"/data2/wl/Part_A_ME_clip/frame",
                                               v_test_path=r"/data2/wl/Part_A_ME_clip/frame",
                                               e_train_path=r"/data3/wl/eye_face/casme3/eye_data2",
                                               e_test_path=r"/data3/wl/eye_face/casme3/eye_data2",
                                               train_label_path=r"/data3/wl/eye_face/casme3/filtered_train_data.txt",
                                               test_label_path=r"/data3/wl/eye_face/casme3/filtered_test_data1.txt",
                                               batch_size=args.batch_size)
   

    val_acc, val_los,war,uar,uF1 = validate(val_loader, model, criterion, args, log_txt_path)
        
    # last_uar, last_war = computer_uar_war(val_loader, model, checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set, args.class_names)
    # best_uar, best_war = computer_uar_war(val_loader, model, best_checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set, args.class_names)
    # print(best_uar, best_war)
    return war,uar,uF1


from sklearn.metrics import recall_score
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

def validate(val_loader, model, criterion, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ',
                             log_txt_path=log_txt_path)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (images, target, audio, name) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            audio = audio.cuda()

            loss = 0
            output = model(images, audio, name)
            weights = [0.0001,0.0002,0.0004,0.0008,0.001,0.002,0.004,0.008,0.01,0.02,0.04,1.0]

            for j in range(len(output)):
                weighted_loss = weights[j] * criterion(output[j], target) if j < len(weights) else criterion(output[j], target)
                loss += weighted_loss
            
            acc1, _ = accuracy(output[-1], target, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            all_preds.append(output[-1].argmax(dim=1).cpu().numpy())
            all_targets.append(target.cpu().numpy())

            if i % args.print_freq == 0:
                progress.display(i)

        # Convert lists to numpy arrays
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Calculate per-class precision, recall, F1
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)

        # Calculate UF1 (unweighted F1 score)
        UF1 = np.mean(f1_per_class)  # Averaging F1 scores for each class

        # Calculate UAR (Unweighted Average Recall) and WAR (Weighted Average Recall)
        uar = recall_score(all_targets, all_preds, average='macro')
        war = recall_score(all_targets, all_preds, average='weighted')

        print(f'Current Accuracy: {top1.avg:.3f}')
        print(f'UAR: {uar:.3f}, WAR: {war:.3f}, UF1: {UF1:.3f}')
        with open(log_txt_path, 'a') as f:
            f.write(f'Current Accuracy: {top1.avg:.3f}, UAR: {uar:.3f}, WAR: {war:.3f}, UF1: {UF1:.3f}\n')

    return top1.avg, losses.avg, uar, war, UF1


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # print(output.shape, target.shape)
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # print(output.shape, target.shape, pred.shape)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

def computer_uar_war(val_loader, model, checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set, class_names):
    
    pre_trained_dict = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(pre_trained_dict)
    
    model.eval()

    correct = 0
    with torch.no_grad():
        for i, (images, target, audio,name) in enumerate(tqdm.tqdm(val_loader)):
            
            images = images.cuda()
            target = target.cuda()
            audio = audio.cuda()
            output = model(images, audio,name)        

            predicted = output[-1].argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            if i == 0:
                all_predicted = predicted
                all_targets = target
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, target), 0)

    war = 100. * correct / len(val_loader.dataset)
    
    # Compute confusion matrix
    _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=4)
    normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag.mean()
        
    print("Confusion Matrix Diag:", list_diag)
    print("UAR: %0.2f" % uar)
    print("WAR: %0.2f" % war)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))

    if args.dataset == "DFEW":
        title_ = "Confusion Matrix on DFEW fold "+str(data_set)
    elif args.dataset == "MAFW":
        title_ = "Confusion Matrix on MAFW fold "+str(data_set)
    elif args.dataset == "EMER":
        title_ = "Confusion Matrix on EMER fold "+str(data_set)

    plot_confusion_matrix(normalized_cm, classes=class_names, normalize=True, title="title_")
    plt.savefig(os.path.join(log_confusion_matrix_path))
    plt.close()
    
    with open(log_txt_path, 'a') as f:
        f.write('************************' + '\n')
        f.write(checkpoint_path)
        f.write("Confusion Matrix Diag:" + '\n')
        f.write(str(list_diag.tolist()) + '\n')
        f.write('UAR: {:.2f}'.format(uar) + '\n')        
        f.write('WAR: {:.2f}'.format(war) + '\n')
        f.write('************************' + '\n')
    
    return uar, war


if __name__ == '__main__':
    args = parse_args() 
    now = datetime.datetime.now()
    time_str = now.strftime("%y%m%d%H%M")
    time_str = time_str + args.exper_name

    print('************************')
    for k, v in vars(args).items():
        print(k,'=',v)
    print('************************')

    if args.dataset == "CASME":
        args.number_class = 3
        args.class_names = [
          "0", "1", '2'
        ]
    uar, war, uF1 = main(1, args)
