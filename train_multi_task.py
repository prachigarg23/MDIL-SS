'''
Multi-task baseline - Joint, Offline learning on the tasks (non-incremental)

1. CS, BDD
2. CS, IDD
3. BDD, IDD
4. CS, BDD, IDD

all encoder weights = shared
all decoders = Domain specific

Training notes:
 - optimizer of shared encoder weights = divide lr by number of tasks, for each iteration/epoch, run forward pass and backprop over all tasks.
 - optimizer of DS weights will have normal lr.
 - all weights are trainable.
 - shared weights are getting updated nb_tasks times in each iter/epoch. all DS weights are getting updated only once for each domain.

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

from iouEval import iouEval, getColorEntry

from models.erfnet_multi_task import Net as Net_MT

from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

NUM_CHANNELS = 3
# default value given, will be overwritten by args.num_classes #cityscapes=20, IDD=27, BDD=20 (same as cityscapes)
NUM_CLASSES = 20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()
current_task = 0  # global inside train


class MyCoTransform(object):
    def __init__(self, augment=True, height=512, width=1024, num_cls=20):
        self.augment = augment
        self.height = height
        self.width = width
        self.num_cls = num_cls
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
        # print('relabeling 255 as: ', self.num_cls-1)

        target = Relabel(255, NUM_CLASSES - 1)(target)

        return input, target


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def is_shared(n): return 'encoder' in n  # all encoder wts are shared


def is_DS_curr(n): return 'decoder' in n  # all decoders are DS


def train(args, model):
    global NUM_CLASSES
    print('datasets: ', args.datasets)
    print('nb_tasks: ', args.nb_tasks)
    print('dataset_name: ', args.dataset)
    print('num_classes: ', args.num_classes)

    best_acc = 0

    tf_dir = 'runs_{}_{}_{}_{}{}_step{}'.format(
        args.dataset, args.model, args.num_epochs, args.batch_size, args.model_name_suffix, len(args.num_classes))
    writer = SummaryWriter('Adaptations/' + tf_dir)

    weight_IDD = torch.tensor([3.235635601598852, 6.76221624390441, 9.458242359884549, 9.446818215454014, 9.947040673126763, 9.789672819856547, 9.476665808564432, 10.465565126694731, 9.59189547383129, 7.637805282159825, 8.990899026692638, 9.26222234098628, 10.265657138809514,
                               9.386517631614392, 8.357391489170013, 9.910382864314824, 10.389977663948363, 8.997422571963602, 10.418070541191673, 10.483262606962834, 9.511436923349441, 7.597725385711079, 6.1734896019878205, 9.787631041755187, 3.9178330193378708, 4.417448652936843, 10.313160683418731])

    weight_BDD = torch.tensor([3.6525147483016243, 8.799815287822142, 4.781908267406055, 10.034828238618045, 9.5567865464289, 9.645099012085169, 10.315292989325766, 10.163473632969513, 4.791692009441432, 9.556915153488912,
                               4.142994047786311, 10.246903827488143, 10.47145010979545, 6.006704177894196, 9.60620532303246, 9.964959813857726, 10.478333987902301, 10.468010534454706, 10.440929141422366, 3.960822533003462])

    weight_city = torch.tensor([2.8159904084894922, 6.9874672455551075, 3.7901719017455604, 9.94305485286704, 9.77037625072462, 9.511470001589007, 10.310780572569994, 10.025305236316246, 4.6341256102158805, 9.561389195953845,
                                7.869695292372276, 9.518873463871952, 10.374050047877898, 6.662394711556909, 10.26054487392723, 10.28786101490449, 10.289883605859952, 10.405463349170795, 10.138502340710136, 5.131658171724055])

    weight_city[19] = 0
    weight_BDD[19] = 0
    weight_IDD[26] = 0

    if args.cuda:
        weight_IDD = weight_IDD.cuda()
        weight_city = weight_city.cuda()
        weight_BDD = weight_BDD.cuda()

    co_transform = MyCoTransform(augment=True, height=args.height, width=args.width)
    co_transform_val = MyCoTransform(augment=False, height=args.height, width=args.width)
    co_transform_idd = MyCoTransform(augment=True, height=args.height, width=args.width, num_cls=27)
    co_transform_val_idd = MyCoTransform(
        augment=False, height=args.height, width=args.width, num_cls=27)

    CS_datadir = '/ssd_scratch/cvit/prachigarg/cityscapes/'
    BDD_datadir = '/ssd_scratch/cvit/prachigarg/bdd100k/seg/'
    IDD_datadir = '/ssd_scratch/cvit/prachigarg/IDD_Segmentation/'

    dataset_train = {}
    dataset_val = {}
    ce_loss = {}

    for data_name in args.datasets:
        if data_name == 'CS':
            print('taking CS')
            dataset_train['CS'] = cityscapes(CS_datadir, co_transform, 'train')
            dataset_val['CS'] = cityscapes(CS_datadir, co_transform_val, 'val')
            ce_loss['CS'] = CrossEntropyLoss2d(weight_city)

        elif data_name == 'BDD':
            print('taking BDD')
            dataset_train['BDD'] = BDD100k(BDD_datadir, co_transform, 'train')
            dataset_val['BDD'] = BDD100k(BDD_datadir, co_transform_val, 'val')
            ce_loss['BDD'] = CrossEntropyLoss2d(weight_BDD)

        elif data_name == 'IDD':
            print('taking IDD')
            dataset_train['IDD'] = IDD(IDD_datadir, co_transform_idd, 'train')
            dataset_val['IDD'] = IDD(IDD_datadir, co_transform_val_idd, 'val')
            ce_loss['IDD'] = CrossEntropyLoss2d(weight_IDD)

    print('\ndataset_train: ', dataset_train)
    print('\ndataset_val: ', dataset_val)
    print('\nce_loss: ', ce_loss)

    loader_train = {dname: DataLoader(dataset_train[dname], num_workers=args.num_workers, batch_size=args.batch_size,
                                      shuffle=True) for dname in args.datasets}
    loader_val = {dname: DataLoader(dataset_val[dname], num_workers=args.num_workers, batch_size=2,
                                    shuffle=True, drop_last=True) for dname in args.datasets}

    # print('global current_task: ', current_task)

    # print('\n\n\n')
    # for name, m in model.named_parameters():
    #     print(name, m.requires_grad)

    savedir = f'../save/{args.savedir}'

    automated_log_path = savedir + "/automated_log.txt"
    modeltxtpath = savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    # define model dicts: separate for parallel_conv_1.1. and parallel_conv_2.1. params and separate for the rest of them.

    params = list(model.named_parameters())
    print('\nusing learning rate this for the W_s params', 5e-4/args.nb_tasks, '\n')
    print('using 5e-4 lr for W_t')

    grouped_parameters = [
        # only the shared conv layers in the encoder will use this lr
        {"params": [p for n, p in params if is_shared(n)], 'lr': 5e-4/args.nb_tasks},
        {"params": [p for n, p in params if is_DS_curr(n)]},  # is domain-specific to current domain
    ]

    optimizer = Adam(
        grouped_parameters, 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=1e-4
    )

    def lambda1(epoch): return pow((1-((epoch-1)/args.num_epochs)), 0.9)  # scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  # scheduler 2

    start_epoch = 1

    n_iters = min([len(loader_train[d]) for d in args.datasets])
    print('n_iters ', n_iters)

    for epoch in range(start_epoch, args.num_epochs+1):
        print("-----TRAINING - EPOCH---", epoch, "-----")

        scheduler.step(epoch)

        epoch_loss = {d: [] for d in args.datasets}
        time_train = []

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        iterator = {dname: iter(loader_train[dname]) for dname in args.datasets}

        model.train()
        for itr in range(n_iters):
            for ind, d in enumerate(args.datasets):
                NUM_CLASSES = args.num_classes[ind]
                images, labels = next(iterator[d])

                if epoch == start_epoch and itr == 1:
                    print('labels are: ', np.unique(labels.numpy()))

                start_time = time.time()
                if args.cuda:
                    inputs = images.cuda()
                    targets = labels.cuda()

                outputs = model(inputs, ind)

                optimizer.zero_grad()
                loss = ce_loss[d](outputs, targets[:, 0])
                loss.backward()
                optimizer.step()

                epoch_loss[d].append(loss.item())
            time_train.append(time.time() - start_time)

        average_epoch_loss_train = {d: np.mean(epoch_loss[d]) for d in args.datasets}
        print('epoch took: ', sum(time_train))

        ############TRAINING OVER##############

        ############VALIDATE ALL DATASETS CONCERNED EVERY 5-10 EPOCHS##########
        # THERE IS NO CURRENT TASK

        # we can do this dataset wise independently for each dataset.
        average_loss_val = {d: 0.0 for d in args.datasets}
        val_acc = {d: 0.0 for d in args.datasets}

        if epoch % 5 == 0 or epoch == 1:
            for ind, d in enumerate(args.datasets):
                print('validate: ', d)
                val_loader = loader_val[d]
                criterion = ce_loss[d]
                evalon_task = ind

                average_loss_val[d], val_acc[d] = eval(
                    model, val_loader, criterion, evalon_task, args.num_classes, epoch)

        # logging tensorboard plots - epoch wise loss and accuracy. Not calculating iouTrain as that will slow down training
        info = {}
        for d in args.datasets:
            k = 'val_acc_{}'.format(d)
            info[k] = val_acc[d]
            k2 = 'val_loss_{}'.format(d)
            info[k2] = average_loss_val[d]
            k3 = 'train_loss_{}'.format(d)
            info[k3] = average_epoch_loss_train[d]
        print(info)

        for tag, value in info.items():
            writer.add_scalar(tag, value, epoch)

        # remember best valIoU and save checkpoint
        temp_acc = sum([val_acc[key] for key in args.datasets])
        if temp_acc == 0:
            current_acc = -0.0
        else:
            current_acc = temp_acc/len(args.datasets)  # Average of the IoUs to save best model

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

    return(model)


def eval(model, dataset_loader, criterion, task, num_classes, epoch):
    # Validate on 500 val images after each epoch of training
    global NUM_CLASSES
    model.eval()
    epoch_loss_val = []
    time_val = []
    num_cls = num_classes[task]
    NUM_CLASSES = num_cls

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
    global current_task
    current_task = args.current_task

    print('\ndataset: ', args.dataset)
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    # Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    print(args.num_classes, args.nb_tasks, args.dataset)

    elif args.model == 'erfnet_multi_task':
        model = Net_MT(args.num_classes, args.nb_tasks, args.current_task)

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.state:
        # init imagenet pretrained enc for erfnet using this function.
        saved_model = torch.load(args.state)

        new_dict_load = {}
        print('loading ImageNet pre-trained enc')
        # only imagenet encoder was saved like module.features.encoder. rest all don't need name changing
        for k, v in saved_model['state_dict'].items():
            nkey = re.sub("module.features", "module", k)
            new_dict_load[nkey] = v

        model.load_state_dict(new_dict_load, strict=False)

    print('loaded\n')

    model = train(args, model)
    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    # NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', default="erfnet_multi_task")
    parser.add_argument('--dataset', default="CSBDD")
    parser.add_argument('--datasets', nargs="+", required=True, default=['CS', 'BDD'])
    parser.add_argument('--dlr', type=float, default=100.0)

    # 27 for level 3 of IDD, 20 for BDD and city
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
