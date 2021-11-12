# Main code for training ERFNet model in Cityscapes dataset
# Sept 2017
# Eduardo Romera
#######################
# individually loads all 3 datasets and handles them separately
# in all ICL models, at 1 time only 1 dataset will be trained. but testing/val will be done on all datasets

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

from models.erfnet_ftp1 import Net as Net_ftp1

from iouEval import iouEval, getColorEntry

from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

NUM_CHANNELS = 3
# default value given, will be overwritten by args.num_classes #cityscapes=20, IDD=27, BDD=20 (same as cityscapes)
NUM_CLASSES_old = 20
NUM_CLASSES_new = 20
NUM_CLASSES = 27  # this is to keep check on background relabeling inside MyCoTransform()

color_transform = Colorize(NUM_CLASSES_new)  # to be modified
image_transform = ToPILImage()


class MyCoTransform(object):
    def __init__(self, augment=True, height=512, width=1024):
        # self.enc = enc
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
        target = Relabel(255, NUM_CLASSES-1)(target)

        return input, target


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


'''
finetune = False : freeze encoder and old decoders, train  only new decoder (FEATURE EXTRACTION)
finetune = True : freeze only old decoders, train new decoder + shared encoder (FINETUNING)

BN - in this file,
decoder_new : will update BN
decoder_old : will not update BN in training
encoder : will update BN in training FE and FT experiments.
'''


def train(args, finetune=False):
    global NUM_CLASSES
    best_acc = 0

    tf_dir = 'runs_{}_{}_{}{}'.format(
        args.model, args.num_epochs, args.batch_size, args.model_name_suffix)
    writer = SummaryWriter('Finetuning_Baselines/' + tf_dir)

    # we are training only on new dataset.
    # VALIDATING on old datasets.
    print('old dataset: ', args.dataset_old)
    print('new dataset: ', args.dataset_new)

    weight_IDD = torch.tensor([3.235635601598852, 6.76221624390441, 9.458242359884549, 9.446818215454014, 9.947040673126763, 9.789672819856547, 9.476665808564432, 10.465565126694731, 9.59189547383129,
                               7.637805282159825, 8.990899026692638, 9.26222234098628, 10.265657138809514, 9.386517631614392, 8.357391489170013, 9.910382864314824, 10.389977663948363, 8.997422571963602,
                               10.418070541191673, 10.483262606962834, 9.511436923349441, 7.597725385711079, 6.1734896019878205, 9.787631041755187, 3.9178330193378708, 4.417448652936843, 10.313160683418731])

    weight_BDD = torch.tensor([3.6525147483016243, 8.799815287822142, 4.781908267406055, 10.034828238618045, 9.5567865464289, 9.645099012085169, 10.315292989325766, 10.163473632969513, 4.791692009441432,
                               9.556915153488912, 4.142994047786311, 10.246903827488143, 10.47145010979545, 6.006704177894196, 9.60620532303246, 9.964959813857726, 10.478333987902301, 10.468010534454706,
                               10.440929141422366, 3.960822533003462])

    weight_city = torch.tensor([2.8159904084894922, 6.9874672455551075, 3.7901719017455604, 9.94305485286704, 9.77037625072462, 9.511470001589007, 10.310780572569994, 10.025305236316246, 4.6341256102158805,
                                9.561389195953845, 7.869695292372276, 9.518873463871952, 10.374050047877898, 6.662394711556909, 10.26054487392723, 10.28786101490449, 10.289883605859952, 10.405463349170795,
                                10.138502340710136, 5.131658171724055])

    # weight[NUM_CLASSES_new-1] = 0
    weight_city[19] = 0
    weight_BDD[19] = 0
    weight_IDD[26] = 0

    co_transform = MyCoTransform(augment=True, height=args.height, width=args.width)  # 1024)
    co_transform_val = MyCoTransform(augment=False, height=args.height, width=args.width)  # 1024)

    dataset_cs_val = cityscapes('/ssd_scratch/cvit/prachigarg/cityscapes/', co_transform_val, 'val')
    dataset_cs_train = cityscapes('/ssd_scratch/cvit/prachigarg/cityscapes/', co_transform, 'train')
    dataset_bdd_val = BDD100k('/ssd_scratch/cvit/prachigarg/bdd100k/seg/', co_transform_val, 'val')
    dataset_bdd_train = BDD100k('/ssd_scratch/cvit/prachigarg/bdd100k/seg/', co_transform, 'train')
    dataset_idd_val = IDD('/ssd_scratch/cvit/prachigarg/IDD_Segmentation/', co_transform_val, 'val')
    dataset_idd_train = IDD('/ssd_scratch/cvit/prachigarg/IDD_Segmentation/', co_transform, 'train')

    if args.dataset_old == 'cityscapes':
        print('loading CS as validation dataset, (old - step 1)')
        loader_val_old = DataLoader(dataset_cs_val, num_workers=args.num_workers,
                                    batch_size=args.batch_size, shuffle=False)
        weight_old = weight_city

    elif args.dataset_old == 'BDD':
        print('loading BDD as validation dataset, (old - step 1)')
        loader_val_old = DataLoader(dataset_bdd_val, num_workers=args.num_workers,
                                    batch_size=args.batch_size, shuffle=False)
        weight_old = weight_BDD

    elif args.dataset_old == 'IDD':
        print('loading IDD as validation dataset, (old - step 1)')
        loader_val_old = DataLoader(dataset_idd_val, num_workers=args.num_workers,
                                    batch_size=args.batch_size, shuffle=False)
        weight_old = weight_IDD

    if args.dataset_new == 'cityscapes':
        print('loading CS as train, val datasets (current step, new dataset)')
        loader = DataLoader(dataset_cs_train, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=True)
        loader_val = DataLoader(dataset_cs_val, num_workers=args.num_workers,
                                batch_size=args.batch_size, shuffle=False)
        weight = weight_city

    elif args.dataset_new == 'BDD':
        print('loading BDD as train, val datasets (current step, new dataset)')
        loader = DataLoader(dataset_bdd_train, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=True)
        loader_val = DataLoader(dataset_bdd_val, num_workers=args.num_workers,
                                batch_size=args.batch_size, shuffle=False)
        weight = weight_BDD

    elif args.dataset_new == 'IDD':
        print('loading IDD as train, val datasets (current step, new dataset)')
        loader = DataLoader(dataset_idd_train, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=True)
        loader_val = DataLoader(dataset_idd_val, num_workers=args.num_workers,
                                batch_size=args.batch_size, shuffle=False)
        weight = weight_IDD

    if args.cuda:
        weight = weight.cuda()
        weight_old = weight_old.cuda()

    criterion = CrossEntropyLoss2d(weight)
    criterion_old = CrossEntropyLoss2d(weight_old)  # to validate the old (step 1) dataset.
    print(type(criterion))
    print('old weights: ', weight_old)
    print('new weights: ', weight)

    savedir = f'../save/{args.savedir}'
    automated_log_path = savedir + "/automated_log.txt"
    modeltxtpath = savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    # Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model = Net_ftp1(NUM_CLASSES_old, NUM_CLASSES_new)

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if args.state:
        print('\check, saved model keys\n')
        saved_model = torch.load(args.state)
        new_dict_load = {}

        for k, v in saved_model['state_dict'].items():
            # print(k)
            nkey = re.sub("decoder", "decoder_old", k)
            new_dict_load[nkey] = v
        model.load_state_dict(new_dict_load, strict=False)

        print('\nLOADED SAVED CITYSCAPES ENC -> ENC, DECODER --> OLD_DECODER for finetuning multi-head model on {}\n'.format(args.dataset_new))
        # got the saved model loaded into defined model. (encoder + old decoder weights get picked up from saved model)
        # weights actually getting loaded or not checking remains
    print('args.finetune: ', args.finetune)

    for name, m in model.named_parameters():
        if 'decoder_old' in name:
            m.requires_grad = False

    finetune_params = list(model.module.encoder.parameters()) + \
        list(model.module.decoder_new.parameters())

    if finetune:
        print('finetuning optimizer')
        optimizer = Adam(finetune_params, 5e-4, (0.9, 0.999),
                         eps=1e-08, weight_decay=1e-4)
    else:
        print('non-finetuning optimizer')
        optimizer = Adam(model.module.decoder_new.parameters(), 5e-4, (0.9, 0.999),
                         eps=1e-08, weight_decay=1e-4)  # feature extraction but BN of shared encoder gets updated

    print('\n\n')
    for name, m in model.named_parameters():
        print(name, m.requires_grad)

    start_epoch = 1

    def lambda1(epoch): return pow((1-((epoch-1)/args.num_epochs)), 0.9)  # scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  # scheduler 2

    for epoch in range(start_epoch, args.num_epochs+1):

        print("----- TRAINING - EPOCH", epoch, "-----")
        scheduler.step(epoch)
        NUM_CLASSES = NUM_CLASSES_new

        epoch_loss = []
        time_train = []

        doIouTrain = args.iouTrain

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES_new, NUM_CLASSES_new-1)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()  # the encoder is allowed to update BN running vars acc to new dataset. but old decoder BN should not get updated
        # model.module.decoder_old.eval()
        # images, labels is of the dataset that is being trained on in this ICL experiment -
        # taking cityscapes pre-trained ERFNet and training BDD -decoder head + share encoder in a fine-tuning setting
        for step, (images, labels) in enumerate(loader):
            if epoch == start_epoch and step == 1:
                print('image size new: ', images.size())
                print('labels size new: ', labels.size())
                print('labels are: ', np.unique(labels.numpy()))
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs, decoder_old=False, decoder_new=True)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()

            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

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

        print("----- VALIDATING - EPOCH", epoch, "--current---")
        average_epoch_loss_val_new, iouVal_new = eval(
            model, loader_val, criterion, NUM_CLASSES_new, epoch, task=1)

        print("----- VALIDATING - EPOCH", epoch, "--old----")
        # validating on old task dataset - cityscapes
        average_epoch_loss_val_old, iouVal_old = eval(
            model, loader_val_old, criterion_old, NUM_CLASSES_old, epoch, task=0)

        # logging tensorboard plots - epoch wise loss and accuracy. Not calculating iouTrain as that will slow down training
        info = {'train_loss': average_epoch_loss_train,
                'val_loss_{}'.format(args.dataset_new): average_epoch_loss_val_new, 'val_accuracy_{}'.format(args.dataset_new): iouVal_new,
                'val_loss_{}'.format(args.dataset_old): average_epoch_loss_val_old, 'val_accuracy_{}'.format(args.dataset_old): iouVal_old}

        for tag, value in info.items():
            writer.add_scalar(tag, value, epoch)

        # remember best valIoU and save checkpoint
        if iouVal_new == 0:  # find best acc to new dataset/task
            current_acc = -average_epoch_loss_val_new
        else:
            current_acc = iouVal_new
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)

        filenameCheckpoint = savedir + \
            '/checkpoint_{}_{}_{}_{}.pth.tar'.format(args.model,
                                                     args.num_epochs, args.batch_size, args.model_name_suffix)
        filenameBest = savedir + \
            '/model_best_{}_{}_{}_{}.pth.tar'.format(args.model,
                                                     args.num_epochs, args.batch_size, args.model_name_suffix)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        # save checkpoint every 50 epochs
        # if epoch % 100 == 0:
        #     filenameCheckpoint50 = savedir + \
        #         '/checkpoint_{}_{}_{}_{}.pth.tar'.format(args.model,
        #                                                  epoch+1, args.batch_size, args.model_name_suffix)
        #
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': str(model),
        #         'state_dict': model.state_dict(),
        #         'best_acc': best_acc,
        #         'optimizer': optimizer.state_dict(),
        #     }, False, filenameCheckpoint50, filenameBest)

        if (is_best):
            with open(savedir + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal_new))

        # SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        # Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (
                epoch, average_epoch_loss_train, average_epoch_loss_val_new, average_epoch_loss_val_old, iouTrain, iouVal_new, iouVal_old, usedLr))

    return(model)  # return model (convenience for encoder-decoder training)


def eval(model, dataset_loader, criterion, num_classes, epoch, task=1):
    # Validate on 500 val images after each epoch of training
    global NUM_CLASSES
    model.eval()
    epoch_loss_val = []
    time_val = []
    NUM_CLASSES = num_classes

    iouEvalVal = iouEval(num_classes, num_classes-1)
    if task == 1:
        decoder_old = False
        decoder_new = True
    elif task == 0:
        decoder_old = True
        decoder_new = False

    print('num_classes: ', NUM_CLASSES, 'decoder_old: ', decoder_old, 'decoder_new: ', decoder_new)

    with torch.no_grad():
        for step, (images, labels) in enumerate(dataset_loader):
            start_time = time.time()
            inputs = images.cuda()
            targets = labels.cuda()

            outputs = model(inputs, decoder_old, decoder_new)
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
    global NUM_CLASSES_old
    global NUM_CLASSES_new
    NUM_CLASSES_old = args.num_classes_old  # this will become a list
    NUM_CLASSES_new = args.num_classes_new
    # print('\ndataset old: ', args.dataset_old)
    # print('\ndataset new: ', args.dataset_new)
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    print("====== FINETUNING TRAINING OF NEW_DECODER & SHARED ENCODER ========")
    model = train(args, args.finetune)  # Train model
    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    # NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', default="erfnet_ftp1")
    parser.add_argument('--dataset-old', default="cityscapes")
    parser.add_argument('--dataset-new', default="BDD")
    # 27 for level 3 of IDD, 20 for BDD and city
    # cityscapes for now. to be converted into list of old num classes
    parser.add_argument('--num-classes-old', type=int, default=20)
    parser.add_argument('--num-classes-new', type=int, default=20)
    parser.add_argument('--state')
    parser.add_argument('--finetune', action='store_true')

    parser.add_argument('--port', type=int, default=8097)
    # parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
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
    parser.add_argument('--model-name-suffix', default="Finetune-CStoBDD-final")

    main(parser.parse_args())
