#!/bin/bash
#SBATCH -A cvit_mobility
#SBATCH -n 14
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=1G
#SBATCH --time=50:00:00
#SBATCH --mail-user=prachigarg2398@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=R-FT-BDD1-%j.out

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
  copy dataset to ssd_scratch
 scp -r prachigarg@ada.iiit.ac.in:/share3/cvit_mobility/prachigarg/IDD/IDD_Segmentation /ssd_scratch/cvit/prachigarg/
 echo "transferred IDD part1 dataset"
 export PYTHONPATH='/home2/prachigarg/public-code/helpers/'
 python /home2/prachigarg/public-code/preperation/createLabels.py --datadir /ssd_scratch/cvit/prachigarg/IDD_Segmentation/ --id-type level3Id
 echo "converted polygon labels to seg masks. preprocessing done"
fi
ls /ssd_scratch/cvit/prachigarg/IDD_Segmentation/gtFine/train/0


# step 2
echo "-----STEP 2: TASK 2: CS -> BDD: FINE-TUNING TRAINING ------------"
echo "--training with IDD pretrained encoder+old decoder head for 150 epochs on 2 GPUs using a bs of 6, new decoder head & shared encoder gets trained on Cityscapes---"
python main_ftp1_enc_newbn.py --savedir /baselines/CS1_BDD2_FT --dataset-old 'cityscapes' --dataset-new 'BDD' --num-classes-old 20 --num-classes-new 20 --num-epochs 150 --batch-size 6 --state "../save/baselines/cityscapes_final_baseline_donot_modify/model_best_cityscapes_prenc.pth.tar" --model-name-suffix="FT_CS1_BDD2" --finetune


# step 3
echo "-----STEP 3: TASK 3: CS->BDD->IDD: FINE-TUNING TRAINING ------------"
python main_FT2_flexible_new.py --savedir /baselines/CS1_BDD2_IDD3_FT --num-epochs 150 --batch-size 6 --state "../save/baselines/CS1_BDD2_FT/checkpoint_erfnet_ftp1_150_6_FT_CS1_BDD2.pth.tar" --model-name-suffix="FT_CS1_BDD2_IDD3" --datasets 'cityscapes' 'BDD' 'IDD' --num-classes 20 20 27 --current_task=2 --nb_tasks=3 --dataset-new='IDD' --finetune
