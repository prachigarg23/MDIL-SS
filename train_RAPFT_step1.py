'''
Training protocol for 1st dataset as per our algorithm: (This helps reparameterize the architecture in subsequent steps)
-> init encoder with imagenet pretrained weights.
-> add the DS layers in encoder
-> train the entire architecture on 1st step dataset without freezing any layers.

'''
import os
import random
import time
import numpy as np
import torch
import math
import re

from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from dataset import VOC12, cityscapes, IDD, BDD100k
from transform import Relabel, ToLabel, Colorize
import itertools
import config_task

import importlib
from iouEval import iouEval, getColorEntry

from models.erfnet_RA_parallel import Net as Net_RAP
# from erfnet_RA_series import Net as Net_RAS
# from erfnet_RCM import Net as Net_RCM
# from erfnet_bn import Net as Net_BN
# from erfnet_onlyRAP import Net as Net_onlyRAP

from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

NUM_CHANNELS = 3
# default value given, will be overwritten by args.num_classes #cityscapes=20, IDD=27, BDD=20 (same as cityscapes)
NUM_CLASSES = 20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

# Augmentations - different function implemented to perform random augments on both image and target


class MyCoTransform(object):
    def __init__(self, augment=True, height=512, width=1024):
        self.augment = augment
        self.height = height
        self.width = width
        pass

    def __call__(self, input, target):
        input = Resize([self.height, self.width], Image.BILINEAR)(input)
        target = Resize([self.height, self.width], Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            # Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2)
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX, transY, 0, 0), fill=0)
            target = ImageOps.expand(target, border=(transX, transY, 0, 0),
                                     fill=255)  # pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))

        input = ToTensor()(input)
        target = ToLabel()(target)
        # print('relabeling 255 as: ', NUM_CLASSES-1)
        target = Relabel(255, NUM_CLASSES - 1)(target)

        return input, target


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def train(args, model):
    global NUM_CLASSES
    NUM_CLASSES = args.num_classes[args.current_task]
    print('NUM_CLASSES: ', NUM_CLASSES)

    best_acc = 0

    tf_dir = 'runs_{}_{}_{}_{}{}_step{}'.format(
        args.dataset, args.model, args.num_epochs, args.batch_size, args.model_name_suffix, len(args.num_classes))
    writer = SummaryWriter('Adaptations/' + tf_dir)

    data_name = args.dataset

    weight_IDD = torch.tensor([3.235635601598852, 6.76221624390441, 9.458242359884549, 9.446818215454014, 9.947040673126763, 9.789672819856547, 9.476665808564432, 10.465565126694731,
                               9.59189547383129, 7.637805282159825, 8.990899026692638, 9.26222234098628, 10.265657138809514, 9.386517631614392, 8.357391489170013, 9.910382864314824, 10.389977663948363,
                               8.997422571963602, 10.418070541191673, 10.483262606962834, 9.511436923349441, 7.597725385711079, 6.1734896019878205, 9.787631041755187, 3.9178330193378708, 4.417448652936843, 10.313160683418731])

    weight_BDD = torch.tensor([3.6525147483016243, 8.799815287822142, 4.781908267406055, 10.034828238618045, 9.5567865464289, 9.645099012085169, 10.315292989325766, 10.163473632969513, 4.791692009441432,
                               9.556915153488912, 4.142994047786311, 10.246903827488143, 10.47145010979545, 6.006704177894196, 9.60620532303246, 9.964959813857726, 10.478333987902301, 10.468010534454706,
                               10.440929141422366, 3.960822533003462])

    weight_city = torch.tensor([2.8159904084894922, 6.9874672455551075, 3.7901719017455604, 9.94305485286704, 9.77037625072462, 9.511470001589007, 10.310780572569994, 10.025305236316246, 4.6341256102158805,
                                9.561389195953845, 7.869695292372276, 9.518873463871952, 10.374050047877898, 6.662394711556909, 10.26054487392723, 10.28786101490449, 10.289883605859952, 10.405463349170795, 10.138502340710136,
                                5.131658171724055])

    weight_city[19] = 0
    weight_BDD[19] = 0
    weight_IDD[26] = 0

    co_transform = MyCoTransform(augment=True, height=args.height, width=args.width)  # 1024)
    co_transform_val = MyCoTransform(augment=False, height=args.height, width=args.width)  # 1024)

    CS_datadir = '/ssd_scratch/cvit/prachigarg/cityscapes/'
    BDD_datadir = '/ssd_scratch/cvit/prachigarg/bdd100k/seg/'
    IDD_datadir = '/ssd_scratch/cvit/prachigarg/IDD_Segmentation/'

    if data_name == 'cityscapes':
        print('taking CS')
        dataset_train = cityscapes(CS_datadir, co_transform, 'train')
        dataset_val = cityscapes(CS_datadir, co_transform_val, 'val')
        weight = weight_city
    elif data_name == 'IDD':
        print('taking IDD')
        dataset_train = IDD(IDD_datadir, co_transform, 'train')
        dataset_val = IDD(IDD_datadir, co_transform_val, 'val')
        weight = weight_IDD
    if data_name == 'BDD':
        print('taking BDD')
        dataset_train = BDD100k(BDD_datadir, co_transform, 'train')
        dataset_val = BDD100k(BDD_datadir, co_transform_val, 'val')
        weight = weight_BDD

    loader = DataLoader(dataset_train, num_workers=args.num_workers,
                        batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=False)

    # dataset_val_cs = cityscapes(CS_datadir, co_transform_val, 'val')
    # loader_val_cs = DataLoader(dataset_val_cs, num_workers=args.num_workers,
    #                            batch_size=args.batch_size, shuffle=False)

    if args.cuda:
        weight = weight.cuda()
        weight_city = weight_city.cuda()

    criterion_city = CrossEntropyLoss2d(weight_city)

    criterion = CrossEntropyLoss2d(weight)
    print(type(criterion))
    current_task = args.current_task

    '''
    RAP-FT model: freeze only DS parameters of the previous domains. Shared params will be trained.
    Freeze: previous decoders + previous DS 'bn' and 'parallel conv' layers
    '''

    if args.model == 'erfnet_RA_parallel':
        for name, m in model.named_parameters():
            if 'decoder' in name:
                if 'decoder.{}'.format(current_task) in name:
                    m.requires_grad = True
                else:
                    m.requires_grad = False

            elif 'encoder' in name:
                if 'bn' in name or 'parallel_conv' in name:
                    if '.{}.weight'.format(current_task) in name or '.{}.bias'.format(current_task) in name:
                        m.requires_grad = True
                    else:
                        m.requires_grad = False

    if args.model == 'erfnet_onlyRAP':
        for name, m in model.named_parameters():
            if 'decoder' in name:
                if 'decoder.{}'.format(current_task) in name:
                    m.requires_grad = True
                else:
                    m.requires_grad = False

            elif 'encoder' in name:
                if 'parallel_conv' in name:
                    if '.{}.weight'.format(current_task) in name or '.{}.bias'.format(current_task) in name:
                        m.requires_grad = True
                    else:
                        m.requires_grad = False

    elif args.model == 'erfnet_RA_series':
        for name, m in model.named_parameters():
            if 'decoder' in name:
                if 'decoder.{}'.format(current_task) not in name:
                    m.requires_grad = False  # by default its true
            elif 'encoder' in name and 'bn' in name:
                if 'bns_1.{}.'.format(current_task) in name or 'bns_2.{}.'.format(current_task) in name or 'bn_ini.{}.'.format(current_task) in name:
                    m.requires_grad = True
                else:
                    m.requires_grad = False
        # the shared convs have been left True

    elif args.model == 'erfnet_RCM':
        for name, m in model.named_parameters():
            if 'decoder' in name:
                if 'decoder.{}'.format(current_task) not in name:
                    m.requires_grad = False  # by default its True. so all previous task decoder weights will be set tp False
            elif 'encoder' in name:
                if 'bn' in name or 'Wt' in name:
                    if 'bns_1.{}.'.format(current_task) in name or 'bns_2.{}.'.format(current_task) in name or 'bn_ini.{}.'.format(current_task) in name:
                        m.requires_grad = True
                    elif 'Wt_1.{}.'.format(current_task) in name or 'Wt_2.{}.'.format(current_task) in name:
                        m.requires_grad = True
                    else:
                        m.requires_grad = False

    elif args.model == 'erfnet_bn':
        print('hi, freezing acc. to DSBN-only experiment')
        for name, m in model.named_parameters():
            if 'decoder' in name:
                if 'decoder.{}'.format(current_task) not in name:
                    m.requires_grad = False  # by default its true
            elif 'encoder' in name and 'bn' in name:
                if 'bns_1.{}.'.format(current_task) in name or 'bns_2.{}.'.format(current_task) in name or 'bn_ini.{}.'.format(current_task) in name:
                    m.requires_grad = True
                else:
                    m.requires_grad = False

    print('\n\n\n')
    for name, m in model.named_parameters():
        print(name, m.requires_grad)

    savedir = f'../save/{args.savedir}'

    automated_log_path = savedir + "/automated_log.txt"
    modeltxtpath = savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),
                     eps=1e-08, weight_decay=1e-4)  # scheduler 2

    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    def lambda1(epoch): return pow((1-((epoch-1)/args.num_epochs)), 0.9)  # scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  # scheduler 2

    start_epoch = 1

    for epoch in range(start_epoch, args.num_epochs+1):
        print("-----TRAINING - EPOCH---", epoch, "-----")

        scheduler.step(epoch)  # scheduler 2

        epoch_loss = []
        time_train = []

        doIouTrain = args.iouTrain

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(loader):
            if epoch == start_epoch and step == 1:
                print('image size new: ', images.size())
                print('labels size new: ', labels.size())
                print('labels are: ', np.unique(labels.numpy()))
                # writer.add_graph(model(), images.cuda(), True) #not working (Segmentation fault (core dumped))

            start_time = time.time()
            if args.cuda:
                inputs = images.cuda()
                targets = labels.cuda()

            outputs = model(inputs, current_task)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        print('epoch took: ', sum(time_train))

        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print("EPOCH IoU on TRAIN set: ", iouStr, "%")

        print("----- VALIDATING - EPOCH", epoch, "-----")
        average_loss_val, val_acc = eval(
            model, loader_val, criterion, current_task, args.num_classes, epoch)
        # every 10 epochs, check validation accuracy on other datasets:
        # if epoch % 10 == 0:
        #     average_loss_val_bdd, val_acc_bdd = eval(
        #         model, loader_val_cs, criterion_city, 0, args.num_classes, epoch)
        #     print('cityscapes loss and acc: ', average_loss_val_bdd, val_acc_bdd)
        # logging tensorboard plots - epoch wise loss and accuracy. Not calculating iouTrain as that will slow down training
        # info = {'train_loss': average_epoch_loss_train, 'val_loss_{}'.format(
        #     data_name): average_loss_val, 'val_acc_{}'.format(data_name): val_acc}

        info = {'train_loss': average_epoch_loss_train, 'val_loss_{}'.format(
            data_name): average_loss_val, 'val_acc_{}'.format(data_name): val_acc}

        for tag, value in info.items():
            writer.add_scalar(tag, value, epoch)

        # remember best valIoU and save checkpoint
        if val_acc == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = val_acc
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)

        'runs_{}_{}_{}_{}{}_step{}'.format(
            args.dataset, args.model, args.num_epochs, args.batch_size, args.model_name_suffix, len(args.num_classes))

        filenameCheckpoint = savedir + \
            '/checkpoint_{}_{}_{}_{}{}_step{}.pth.tar'.format(
                args.dataset, args.model, args.num_epochs, args.batch_size, args.model_name_suffix, len(args.num_classes))
        filenameBest = savedir + \
            '/model_best_{}_{}_{}_{}{}_step{}.pth.tar'.format(
                args.dataset, args.model, args.num_epochs, args.batch_size, args.model_name_suffix, len(args.num_classes))

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        if (is_best):
            # torch.save(model.state_dict(), filenamebest)
            # print(f'save: {filenamebest} (epoch: {epoch})')
            with open(savedir + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, val_acc))

        # SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        # Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (
                epoch, average_epoch_loss_train, average_loss_val, iouTrain, val_acc, usedLr))

    return(model)


def eval(model, dataset_loader, criterion, task, num_classes, epoch):
    # Validate on 500 val images after each epoch of training

    model.eval()
    epoch_loss_val = []
    time_val = []
    num_cls = num_classes[task]
    print('number of classes in current task: ', num_cls)
    print('validating task: ', task)
    iouEvalVal = iouEval(num_cls, num_cls-1)

    with torch.no_grad():
        for step, (images, labels) in enumerate(dataset_loader):
            start_time = time.time()
            inputs = images.cuda()
            targets = labels.cuda()

            outputs = model(inputs, task)
            if step == 1:
                print('------------------', outputs.size(), targets.size())

            loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)

            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

            if 50 > 0 and step % 50 == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / 6))

    average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

    iouVal = 0
    iouVal, iou_classes = iouEvalVal.getIoU()
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print("EPOCH IoU on VAL set: ", iouStr, "%")
    print('check val fn, loss, acc: ', average_epoch_loss_val, iouVal)
    return average_epoch_loss_val, iouVal


def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    print("Saving model: ", filenameCheckpoint)
    if is_best:
        print("Saving model as best: ", filenameBest)
        torch.save(state, filenameBest)


def main(args):
    print('\ndataset: ', args.dataset)
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    # Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    print(args.num_classes, args.nb_tasks, args.dataset)

    if args.model == 'erfnet_RA_parallel':
        model = Net_RAP(args.num_classes, args.nb_tasks, args.current_task)
    elif args.model == 'erfnet_RA_series':
        model = Net_RAS(args.num_classes, args.nb_tasks, args.current_task)
    elif args.model == 'erfnet_RCM':
        model = Net_RCM(args.num_classes, args.nb_tasks, args.current_task)
    elif args.model == 'erfnet_bn':
        model = Net_BN(args.num_classes, args.nb_tasks, args.current_task)
    elif args.model == 'erfnet_onlyRAP':
        model = Net_onlyRAP(args.num_classes, args.nb_tasks, args.current_task)

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.state:
        # trying to init imagenet pretrained enc for erfnet using this function.
        # model is defined using erfnet_RA_parallel code.
        saved_model = torch.load(args.state)

        # print('\n\ndefined model keys\n')
        # for k in model.state_dict().keys():
        #     print(k)
        #
        # print('\n\nsaved model keys\n')
        # for k in saved_model['state_dict'].keys():
        #     print(k)

        # only for 1st task we will use the imagenet pretrained encoder. rest of the tasks can directly copy whatever params are common from the previous checkpoint.
        if args.current_task == 0:
            print(args.dataset)
            new_dict_load = {}
            print('loading ImageNet pre-trained enc')
            # only imagenet encoder was saved like module.features.encoder. rest all will don't need name changing
            for k, v in saved_model['state_dict'].items():
                nkey = re.sub("module.features", "module", k)
                new_dict_load[nkey] = v

            model.load_state_dict(new_dict_load, strict=False)

        else:
            print('\n\n\ncheck, keys in loaded model that are there in defined model\n')

            for k, v in saved_model['state_dict'].items():
                if k in model.state_dict().keys():
                    print(k)

            print('loading previous step weights')
            model.load_state_dict(saved_model['state_dict'], strict=False)
        print('loaded model from checkpoint provided.')

    print('loaded\n')

    model = train(args, model)
    # print('\nMODEL:\n', model)
    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    # NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', default="erfnet_RA_parallel")  # give erfnet_bn
    parser.add_argument('--dataset', default="cityscapes")

    # 27 for level 3 of IDD, 20 for BDD and city
    # do type=int, nargs='+' when you want to pass as input a list of integers
    parser.add_argument('--num-classes', type=int, nargs="+", help='pass list with number of classes',
                        required=True, default=[20])  # send [20, 20] in IL-step1, [20, 20, 27] in IL-step2
    parser.add_argument('--nb_tasks', type=int, default=1)  # 2 for IL-step1, 3 for IL-step2
    # 0 for IL-step1 (CS), 1 for IL-step2 (BDD), 2 for IL-step3 (IDD)
    parser.add_argument('--current_task', type=int, default=0)
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    # You can use this value to save model every X epochs
    parser.add_argument('--epochs-save', type=int, default=0)
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    # , default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--pretrainedEncoder')

    # recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouTrain', action='store_true', default=False)
    parser.add_argument('--iouVal', action='store_true', default=True)
    # Use this flag to load last checkpoint for training
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model-name-suffix', default="RAP_FT")

    main(parser.parse_args())
