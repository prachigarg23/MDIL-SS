#!/bin/bash
#SBATCH -A cvit_mobility
#SBATCH -n 10
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=1-11:00:00
#SBATCH --mail-user=prachigarg2398@gmail.com
#SBATCH --mail-type=ALL

#activate conda environment
source /home2/prachigarg/anaconda3/etc/profile.d/conda.sh
conda activate erfnet
echo "conda environment activated"

if [ ! -d "/ssd_scratch/cvit/prachigarg/" ]
then
  mkdir /ssd_scratch/cvit/prachigarg
fi

[ -d "/ssd_scratch/cvit/prachigarg/cityscapes" ] && echo "Directory /ssd_scratch/cvit/prachigarg/cityscapes exists."
if [ ! -d "/ssd_scratch/cvit/prachigarg/cityscapes" ]
then
  scp -r prachigarg@ada.iiit.ac.in:/share3/cvit_mobility/prachigarg/cityscapes /ssd_scratch/cvit/prachigarg/
  echo "transferred cityscapes dataset"
fi

[ -d "/ssd_scratch/cvit/prachigarg/bdd100k" ] && echo "Directory /ssd_scratch/cvit/prachigarg/bdd100k exists."
if [ ! -d "/ssd_scratch/cvit/prachigarg/bdd100k" ]
then
  scp -r prachigarg@ada.iiit.ac.in:/share3/cvit_mobility/prachigarg/bdd100k /ssd_scratch/cvit/prachigarg/
  echo "transferred BDD100k dataset"
fi

[ -d "/ssd_scratch/cvit/prachigarg/IDD_Segmentation" ] && echo "Directory /ssd_scratch/cvit/prachigarg/IDD_Segmentation exists."
ls /ssd_scratch/cvit/prachigarg/
if [ ! -d "/ssd_scratch/cvit/prachigarg/IDD_Segmentation" ]
then
  scp -r prachigarg@ada.iiit.ac.in:/share3/cvit_mobility/prachigarg/IDD/IDD_Segmentation /ssd_scratch/cvit/prachigarg/
  echo "transferred IDD part1 dataset"
  export PYTHONPATH='/home2/prachigarg/public-code/helpers/'
  python /home2/prachigarg/public-code/preperation/createLabels.py --datadir /ssd_scratch/cvit/prachigarg/IDD_Segmentation/ --id-type level3Id
  echo "converted polygon labels to seg masks. preprocessing done"
fi

echo "starting training of cityscapes baseline on erfnet. training with imagenet pretrained encoder for 150 epochs on 2 GPUs using a bs of 6"
python main.py --savedir /baselines --datadir /ssd_scratch/cvit/prachigarg/cityscapes --num-epochs 150 --batch-size 6 --decoder --pretrainedEncoder "../trained_models/erfnet_encoder_pretrained.pth.tar" --model-name-suffix="erfnet-cityscapes-base"

echo "starting training of BDD100k baseline on erfnet, image resolution = 1024x512. training with imagenet pretrained encoder for 150 epochs on 2 GPUs using a bs of 6"
python main.py --dataset="BDD" --savedir /baselines/BDD --datadir /ssd_scratch/cvit/prachigarg/bdd100k/seg --num-epochs 150 --batch-size 6 --decoder --pretrainedEncoder "../trained_models/erfnet_encoder_pretrained.pth.tar" --model-name-suffix="erfnet-BDD-base-1024x512" --num-classes 20 --height=512 --width=1024

echo "starting training of IDD baseline on erfnet, image resolution = 1024x512. training with imagenet pretrained encoder for 150 epochs on 2 GPUs using a bs of 6"
python main.py --dataset="IDD" --savedir /baselines/IDDP1 --datadir /ssd_scratch/cvit/prachigarg/IDD_Segmentation --num-epochs 200 --batch-size 6 --decoder --pretrainedEncoder "../trained_models/erfnet_encoder_pretrained.pth.tar" --model-name-suffix="erfnet-IDDP1-base-1024x512" --num-classes 27 --height=512 --width=1024
