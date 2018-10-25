from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from torchreid import data_manager
from torchreid.dataset_loader import ImageDataset
from torchreid import transforms as T
from torchreid import models
from torchreid.losses import CrossEntropyLabelSmooth, DeepSupervision
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.logger import Logger
from torchreid.utils.torchtools import set_bn_to_eval, count_num_param
from torchreid.optimizers import init_optim


parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='data',
                    help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='bot',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=224,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0,
                    help="split index (0-based)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=64, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=64, type=int,
                    help="test batch size")
# 参数调节
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[20, 40], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=1e-6, type=float,
                    help="weight decay (default: 5e-04)")

# Architecture
parser.add_argument('-a', '--arch', type=str, default='ResNet50_BOT_MultiTask', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10,
                    help="print frequency")
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
# 评估模式参数
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--eval-step', type=int, default=1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")

parser.add_argument('--save-dir', type=str, default='log/resnet_50_bot')

# 选择使用哪个GPU
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--vis-ranked-res', action='store_true',
                    help="visualize ranked results, only available in evaluation mode (default: False)")

# global variables
args = parser.parse_args()



def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_imgreid_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id
    )

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    testloader = DataLoader(
        ImageDataset(dataset.test, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, loss={'xent'}, use_gpu=use_gpu)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    gender_criterion_xent = nn.CrossEntropyLoss()
    staff_criterion_xent = nn.CrossEntropyLoss()
    customer_criterion_xent = nn.CrossEntropyLoss()
    stand_criterion_xent = nn.CrossEntropyLoss()
    sit_criterion_xent = nn.CrossEntropyLoss()
    phone_criterion_xent = nn.CrossEntropyLoss()

    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)

    if use_gpu:
        model = nn.DataParallel(model).cuda()


    start_time = time.time()
    train_time = 0
    best_score = 0
    best_epoch = args.start_epoch
    print("==> Start training")

################################### 修改到这里，把train 和 test改一下就好
    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, gender_criterion_xent, staff_criterion_xent, customer_criterion_xent, \
              stand_criterion_xent, sit_criterion_xent, phone_criterion_xent, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)
        
        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            gender_accurary, staff_accurary, customer_accurary, stand_accurary, sit_accurary, phone_accurary = test(model, testloader, use_gpu)
            Score = (gender_accurary + staff_accurary + customer_accurary + stand_accurary + sit_accurary + phone_accurary) * 100
            is_best = Score > best_score

            if is_best:
                best_score = Score
                best_gender_acc = gender_accurary
                best_staff_acc = staff_accurary
                best_customer_acc = customer_accurary
                best_stand_acc = stand_accurary
                best_sit_acc = sit_accurary
                best_phone_acc = phone_accurary
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'rank1': Score,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best best_score {} |Gender_acc {}\t Staff_acc {}\t Customer_acc {}\t Stand_acc {}\t Sit_acc {}\t Phone_acc {}|achieved at epoch {}"
        .format(best_score, best_gender_acc, best_staff_acc, best_customer_acc, best_stand_acc, best_sit_acc,
                best_phone_acc, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, gender_criterion_xent, staff_criterion_xent, customer_criterion_xent, \
          stand_criterion_xent, sit_criterion_xent, phone_criterion_xent, \
          optimizer, trainloader, use_gpu, freeze_bn=False):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, gender_labels, staff_labels, customer_labels, stand_labels, sit_labels,\
                    play_with_phone_labels) in enumerate(trainloader):
        data_time.update(time.time() - end)
#  修改到这里
        if use_gpu:
            imgs, gender_labels, staff_labels, customer_labels, stand_labels, sit_labels, play_with_phone_labels = \
            imgs.cuda(), gender_labels.cuda(), staff_labels.cuda(),\
            customer_labels.cuda(), stand_labels.cuda(), sit_labels.cuda(), play_with_phone_labels.cuda()
        
        gender_outputs, staff_outputs, customer_outputs, stand_outputs, sit_outputs, play_with_phone_outputs = model(imgs)

        gender_loss = gender_criterion_xent(gender_outputs, gender_labels)
        staff_loss = staff_criterion_xent(staff_outputs, staff_labels)
        customer_loss = customer_criterion_xent(customer_outputs, customer_labels)
        stand_loss = stand_criterion_xent(stand_outputs, stand_labels)
        sit_loss = sit_criterion_xent(sit_outputs, sit_labels)
        phone_loss = phone_criterion_xent(play_with_phone_outputs, play_with_phone_labels)

        loss = gender_loss + staff_loss + customer_loss + stand_loss + sit_loss + phone_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), gender_labels.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            print('gender_loss:{:.3f}\t staff_loss:{:.3f}\t customer_loss:{:.3f}\t stand_loss{:.3f}\n '
                  'sit_loss:{:.3f}\t phone_loss:{:.3f}\t '.format(gender_loss, staff_loss, customer_loss, stand_loss,\
                                                            sit_loss, phone_loss))
        
        end = time.time()


def test(model, testloader, use_gpu):
    batch_time = AverageMeter()
    gender_correct = 0
    staff_correct = 0
    customer_correct = 0
    stand_correct = 0
    sit_correct = 0
    phone_correct = 0

    total = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (imgs, gender_labels, staff_labels, customer_labels, stand_labels,\
                        sit_labels, phone_labels) in enumerate(testloader):
            if use_gpu:
                imgs, gender_labels, staff_labels, customer_labels, stand_labels, sit_labels, phone_labels = \
					imgs.cuda(), gender_labels.cuda(), staff_labels.cuda(), customer_labels.cuda(), \
                    stand_labels.cuda(), sit_labels.cuda(), phone_labels.cuda()
            total += gender_labels.size(0)

            gender_outputs, staff_outputs, customer_outputs, stand_outputs, sit_outputs, play_with_phone_outputs = model(
                imgs)

            _, gender_predicted = torch.max(gender_outputs.data, 1)
            _, staff_predicted = torch.max(staff_outputs.data, 1)
            _, customer_predicted = torch.max(customer_outputs.data, 1)
            _, stand_predicted = torch.max(stand_outputs.data, 1)
            _, sit_predicted = torch.max(sit_outputs.data, 1)
            _, phone_predicted = torch.max(play_with_phone_outputs.data, 1)

            gender_correct += (gender_predicted == gender_labels).sum()
            staff_correct += (staff_predicted == staff_labels).sum()
            customer_correct += (customer_predicted == customer_labels).sum()
            stand_correct += (stand_predicted == stand_labels).sum()
            sit_correct += (sit_predicted == sit_labels).sum()
            phone_correct += (phone_predicted == phone_labels).sum()

        gender_correct = gender_correct.cpu().numpy()
        staff_correct = staff_correct.cpu().numpy()
        customer_correct = customer_correct.cpu().numpy()
        stand_correct = stand_correct.cpu().numpy()
        sit_correct = sit_correct.cpu().numpy()
        phone_correct = phone_correct.cpu().numpy()

        gender_accurary = float(gender_correct / total)
        staff_accurary = float(staff_correct / total)
        customer_accurary = float(customer_correct / total)
        stand_accurary = float(stand_correct / total)
        sit_accurary = float(sit_correct / total)
        phone_accurary = float(phone_correct / total)

        print(
            'Accurary:|gender {:.2f}%|\tstaff {:.2f}%|\tcustomer {:.2f}%|\tstand {:.2f}%|\tsit {:.2f}%|\tphone {:.2f}%|'
            .format(gender_accurary * 100, staff_accurary * 100, customer_accurary * 100, stand_accurary * 100,
                    sit_accurary * 100, phone_accurary * 100))

    return gender_accurary,staff_accurary,customer_accurary,stand_accurary,sit_accurary,phone_accurary




if __name__ == '__main__':
    main()
