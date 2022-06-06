#!/bin/bash

#SBATCH -J  unet2D
#SBATCH -o  unet2D.out
#SBATCH -N  1             ### 총 필요한 컴퓨터 노드 수 
#SBATCH -n  1             ### 총 필요한 프로세스 수
#SBATCH -p  intern         ### partition 이름
##SBATCH -t  01:30:00     ### 최대 작업 시간

module purge
module load cuda/cuda-10.2


######## execute ########
echo ''
echo '-----------START-----------'
echo ''
/home/intern1/.conda/envs/BraTS/bin/python -u unet2D.py >> unet2D.out
echo ''
echo '------------END------------'
echo ''