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

from models.erfnet_ft2 import Net as Net_ft2
from iouEval import iouEval, getColorEntry

from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

NUM_CHANNELS = 3

NUM_CLASSES_new = 27
NUM_CLASSES = 20

color_transform = Colorize(NUM_CLASSES)  # to be modified
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

    # WEIGHTS are needed for new class only
    weight_IDD = torch.tensor([3.235635601598852, 6.76221624390441, 9.458242359884549, 9.446818215454014, 9.947040673126763, 9.789672819856547, 9.476665808564432, 10.465565126694731, 9.59189547383129,
                               7.637805282159825, 8.990899026692638, 9.26222234098628, 10.265657138809514, 9.386517631614392, 8.357391489170013, 9.910382864314824, 10.389977663948363, 8.997422571963602,
                               10.418070541191673, 10.483262606962834, 9.511436923349441, 7.597725385711079, 6.1734896019878205, 9.787631041755187, 3.9178330193378708, 4.417448652936843, 10.313160683418731])

    weight_BDD = torch.tensor([3.6525147483016243, 8.799815287822142, 4.781908267406055, 10.034828238618045, 9.5567865464289, 9.645099012085169, 10.315292989325766, 10.163473632969513, 4.791692009441432,
                               9.556915153488912, 4.142994047786311, 10.246903827488143, 10.47145010979545, 6.006704177894196, 9.60620532303246, 9.964959813857726, 10.478333987902301, 10.468010534454706,
                               10.440929141422366, 3.960822533003462])

    weight_city = torch.tensor([2.8159904084894922, 6.9874672455551075, 3.7901719017455604, 9.94305485286704, 9.77037625072462, 9.511470001589007, 10.310780572569994, 10.025305236316246, 4.6341256102158805,
                                9.561389195953845, 7.869695292372276, 9.518873463871952, 10.374050047877898, 6.662394711556909, 10.26054487392723, 10.28786101490449, 10.289883605859952, 10.405463349170795,
                                10.138502340710136, 5.131658171724055])

    weight_city[19] = 0
    weight_BDD[19] = 0
    weight_IDD[26] = 0
    if args.cuda:
        weight_IDD = weight_IDD.cuda()
        weight_BDD = weight_BDD.cuda()
        weight_city = weight_city.cuda()

    ce_loss = {}
    ce_loss['cityscapes'] = CrossEntropyLoss2d(weight_city)
    ce_loss['IDD'] = CrossEntropyLoss2d(weight_IDD)
    ce_loss['BDD'] = CrossEntropyLoss2d(weight_BDD)

    co_transform = MyCoTransform(augment=True, height=args.height, width=args.width)  # 1024)
    co_transform_val = MyCoTransform(augment=False, height=args.height, width=args.width)  # 1024)

    dataset_cs_train = cityscapes('/ssd_scratch/cvit/prachigarg/cityscapes/', co_transform, 'train')
    dataset_bdd_train = BDD100k('/ssd_scratch/cvit/prachigarg/bdd100k/seg/', co_transform, 'train')
    dataset_idd_train = IDD('/ssd_scratch/cvit/prachigarg/IDD_Segmentation/', co_transform, 'train')

    # train_loader, train criterion
    print('loading new data for train')
    if args.dataset_new == 'IDD':
        print('taking IDD')
        loader = DataLoader(dataset_idd_train, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=True)
        criterion = ce_loss['IDD']  # for training loop
    elif args.dataset_new == 'BDD':
        print('taking BDD')
        loader = DataLoader(dataset_bdd_train, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=True)
        criterion = ce_loss['BDD']
    elif args.dataset_new == 'cityscapes':
        print('taking CS')
        loader = DataLoader(dataset_cs_train, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=True)
        criterion = ce_loss['cityscapes']

    dataset_val = {}
    dataset_val['IDD'] = IDD('/ssd_scratch/cvit/prachigarg/IDD_Segmentation/',
                             co_transform_val, 'val')
    dataset_val['BDD'] = BDD100k(
        '/ssd_scratch/cvit/prachigarg/bdd100k/seg/', co_transform_val, 'val')
    dataset_val['cityscapes'] = cityscapes(
        '/ssd_scratch/cvit/prachigarg/cityscapes/', co_transform_val, 'val')

    # eval has to be done on all 3 datasets. we want to automate it, by giving the
    # (1) right ordering of datasets,
    # (2) num_classes
    # (3) dataset to train on
    # (4) checkpoint
    loader_val = {dname: DataLoader(dataset_val[dname], num_workers=args.num_workers, batch_size=args.batch_size,
                                    shuffle=True) for dname in args.datasets}

    savedir = f'../save/{args.savedir}'
    enc = False
    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        # modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        # modeltxtpath = savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    classes = args.num_classes
    # Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model = Net_ft2(classes[0], classes[1], classes[2])

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if args.state:
        print('inside args.state check\n')
        print('\check, saved model keys\n')
        saved_model = torch.load(args.state)
        new_dict_load = {}
        for k, v in saved_model['state_dict'].items():
            if 'decoder_old' in k:
                nkey = re.sub("decoder_old", "decoder_old1", k)  # city decoder
            elif 'decoder_new' in k:
                nkey = re.sub("decoder_new", "decoder_old2", k)  # bdd decoder
            else:
                nkey = k
            new_dict_load[nkey] = v
        model.load_state_dict(new_dict_load, strict=False)

        print('\nLOADED SAVED CS-BDD ENC -> ENC, Dold->D1, Dnew->D2 for finetuning multi-head model on {}\n'.format(args.dataset_new))
        # got the saved model loaded into defined model. (encoder + old decoder weights get picked up from saved model)
        # weights actually getting loaded or not checking remains
    print('args.finetune: ', args.finetune)

    for name, m in model.named_parameters():
        if 'decoder_old1' in name or 'decoder_old2' in name:
            m.requires_grad = False

    finetune_params = list(model.module.encoder.parameters()) + \
        list(model.module.decoder_new.parameters())

    if finetune:
        optimizer = Adam(finetune_params, 5e-4, (0.9, 0.999),
                         eps=1e-08, weight_decay=1e-4)
    else:
        print('hi, defining optimizer in FE mode')
        optimizer = Adam(model.module.decoder_new.parameters(), 5e-4, (0.9, 0.999),
                         eps=1e-08, weight_decay=1e-4)

    start_epoch = 1

    def lambda1(epoch): return pow((1-((epoch-1)/args.num_epochs)), 0.9)  # scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  # scheduler 2

    for epoch in range(start_epoch, args.num_epochs+1):
        NUM_CLASSES = args.num_classes[2]
        print("----- TRAINING - EPOCH", epoch, "-----")
        scheduler.step(epoch)

        epoch_loss = []
        time_train = []

        doIouTrain = args.iouTrain
        doIouVal = args.iouVal

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES, NUM_CLASSES-1)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()

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
            outputs = model(inputs, decoder_old1=False, decoder_old2=False, decoder_new=True)

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

        average_loss_val = {d: 0.0 for d in args.datasets}
        val_acc = {d: 0.0 for d in args.datasets}

        if epoch % 10 == 0 or epoch == 1:
            print("----- VALIDATING - EPOCH", epoch)
            for ind, d in enumerate(args.datasets):
                print('validate: ', d)
                average_loss_val[d], val_acc[d] = eval(
                    model, loader_val[d], ce_loss[d], args.num_classes[ind], epoch, ind)  # eval(model, dataset_loader, criterion, num_classes, epoch, task=2)

        info = {}
        for d in args.datasets:
            k = 'val_acc_{}'.format(d)
            info[k] = val_acc[d]
            k2 = 'val_loss_{}'.format(d)
            info[k2] = average_loss_val[d]
        print(info)

        for tag, value in info.items():
            writer.add_scalar(tag, value, epoch)

        # remember best valIoU and save checkpoint
        # find best acc to new dataset/task, last index is new dataset.
        if val_acc[args.dataset_new] == 0:
            current_acc = -average_loss_val[args.dataset_new]
        else:
            current_acc = val_acc[args.dataset_new]
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

        if (is_best):
            with open(savedir + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" %
                             (epoch, val_acc[args.dataset_new]))

        # SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        # Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        # with open(automated_log_path, "a") as myfile:
        #     myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (
        #         epoch, average_epoch_loss_train, average_epoch_loss_val_new, iouTrain, val_acc[args.dataset_new], usedLr))

    return(model)


def eval(model, dataset_loader, criterion, num_classes, epoch, task=2):
    # Validate on 500 val images after each epoch of training
    global NUM_CLASSES
    model.eval()
    epoch_loss_val = []
    time_val = []
    NUM_CLASSES = num_classes
    print('inside eval(), dataset_loader: {}, criterion: {}, num_classes: {}, task: {}'.format(
        dataset_loader, criterion, num_classes, task))

    iouEvalVal = iouEval(num_classes, num_classes-1)

    if task == 2:
        decoder_old1 = False
        decoder_old2 = False
        decoder_new = True
    elif task == 1:
        decoder_old1 = False
        decoder_old2 = True
        decoder_new = False
    elif task == 0:
        decoder_old1 = True
        decoder_old2 = False
        decoder_new = False

    print('num_classes: ', NUM_CLASSES, 'decoder_old1: ', decoder_old1,
          'decoder_old2: ', decoder_old2, 'decoder_new: ', decoder_new)

    with torch.no_grad():
        for step, (images, labels) in enumerate(dataset_loader):
            start_time = time.time()
            inputs = images.cuda()
            targets = labels.cuda()

            outputs = model(inputs, decoder_old1, decoder_old2, decoder_new)
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
    global NUM_CLASSES_new

    print('\ndataset old T1: ', args.datasets[0], args.num_classes[0])
    print('\ndataset old T2: ', args.datasets[1], args.num_classes[1])
    print('\ndataset new T3: ', args.datasets[2], args.num_classes[2])
    print('\ndataset_new: ', args.dataset_new)
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
    parser.add_argument('--model', default="erfnet_ftp2")
    parser.add_argument('--dataset-new', default="IDD")
    parser.add_argument('--datasets', nargs="+", help='pass list of datasets in order',
                        required=True, default=['IDD', 'CS', 'BDD'])
    parser.add_argument('--current_task', type=int, default=2)
    parser.add_argument('--nb_tasks', type=int, default=3)
    parser.add_argument('--num-classes', type=int, nargs="+", help='pass list with number of classes in correct order',
                        required=True, default=[20, 20, 27])

    parser.add_argument('--state')
    parser.add_argument('--finetune', action='store_true')

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
    parser.add_argument('--model-name-suffix', default="FE-CSBDDtoIDD-oldencBN")

    main(parser.parse_args())
