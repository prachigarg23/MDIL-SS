#!/bin/bash
#SBATCH -A cvit_mobility
#SBATCH -n 18
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=50:00:00
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
ls /ssd_scratch/cvit/prachigarg/cityscapes/gtFine/train/munich

[ -d "/ssd_scratch/cvit/prachigarg/bdd100k" ] && echo "Directory /ssd_scratch/cvit/prachigarg/bdd100k exists."
if [ ! -d "/ssd_scratch/cvit/prachigarg/bdd100k" ]
then
  scp -r prachigarg@ada.iiit.ac.in:/share3/cvit_mobility/prachigarg/bdd100k /ssd_scratch/cvit/prachigarg/
  echo "transferred BDD100k dataset"
fi
ls /ssd_scratch/cvit/prachigarg/bdd100k/seg/

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
ls /ssd_scratch/cvit/prachigarg/IDD_Segmentation/gtFine/train/0

# joint multi-head model training of CS and IDD datasets
echo "-----starting MULTI-TASK {CS, IDD} Model Training--------------"
python train_multi_task.py --savedir Adaptations/MT/cs-idd --num-epochs 150 --dataset='CSIDD' --batch-size 6 --state "../trained_models/erfnet_encoder_pretrained.pth.tar" --datasets 'CS' 'IDD' --num-classes 20 27 --nb_tasks=2
echo "--done---"

# joint multi-head model training of CS, BDD and IDD datasets
echo "-----starting MULTI-TASK {CS, BDD, IDD} Model Training--------------"
python train_multi_task.py --savedir Adaptations/MT/cs-bdd-idd --num-epochs 150 --batch-size 6 --dataset='CSBDDIDD' --state "../trained_models/erfnet_encoder_pretrained.pth.tar" --datasets 'CS' 'BDD' 'IDD' --num-classes 20 20 27 --nb_tasks=3
echo "--done---"
