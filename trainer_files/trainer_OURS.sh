#!/bin/bash
#SBATCH -A cvit_mobility
#SBATCH -n 18
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=98:00:00
#SBATCH --mail-user=prachigarg2398@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=R-Ours-CSBDDIDD-allsteps-%j.out

# activate conda environment
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


# step 1================================================================================
echo "-----STEP 1: TASK 1: CS: starting RAP-FT model TRAINING on CITYSCAPES-----------"
echo "--training with DS-RAP units and DS-BN units in encoder and single decoder head for CS. using ERFNet architecture--------"
python train_RAPFT_step1.py --savedir Adaptations/RAP_FT_CS1 --num-epochs 150 --batch-size 6 --state "../trained_models/erfnet_encoder_pretrained.pth.tar" --num-classes 20 --current_task=0 --dataset='cityscapes'
echo "-----done-------"


# step 2 - can take 30-40 hours on 2 Nvidia GeForce GTX 1080 Ti==================
echo "-----STEP 2: TASK 2: CS->BDD: starting RAPFT-KLD (OURS), KLD between {cs_old, cs_curr}, training model on BDD. lambdac=0.1------------"
python train_new_task_step2.py --savedir Adaptations/RAP_FT_KLD/CS1_BDD2 --num-epochs 150 --model-name-suffix='ours-CS1-BDD2' --batch-size 6 --state "../save/Adaptations/RAP_FT_CS1/model_best_cityscapes_erfnet_RA_parallel_150_6RAP_FT_step1.pth.tar" --dataset='BDD' --dataset_old='cityscapes' --num-classes 20 20 --current_task=1 --nb_tasks=2 --num-classes-old 20
echo "--done---"


# step 3 - can take ~96 hours on 4 Nvidia GeForce GTX 1080 Ti====================
echo "-----STEP 3: TASK 3: CS|BDD->IDD: KLD between {cs_old, cs_curr} + {bdd_old, bdd_curr}, training model on IDD. lambdac=0.1--------------"
python train_new_task_step3.py --savedir Adaptations/RAP_FT_KLD/CS1_BDD2_IDD3 --num-epochs 150 --model-name-suffix='OURS-CS1-BDD2-IDD3' --batch-size 6 --state "../save/Adaptations/RAP_FT_KLD/CS1_BDD2/checkpoint_BDD_erfnet_RA_parallel_150_6ours-CS1-BDD2_step2.pth.tar" --dataset-new='IDD' --datasets 'cityscapes' 'BDD' 'IDD' --num-classes 20 20 27 --num-classes-old 20 20 --current_task=2 --nb_tasks=3 --lambdac=0.1
echo "--done---"
