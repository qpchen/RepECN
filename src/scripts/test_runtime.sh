#!/bin/bash

#########################################
# This script is used for test the runtime of single 720P image
# run this script use command following (if just have one size, just input anything):
# ./test_runtime.sh [model_name] [size] [scale] [patch] [times] "[addition options]"
# #### specially for RepECN model ,just input Version, and need addition model size param:
# #### ./test_runtime.sh [version] [size] [scale] [patch] [times] "[addition options]"
#########################################
# after copy ../runtime_models/*.pt files and ../../dataset/runtime dir, run example like: 
# ./scripts/test_runtime.sh ALAN v1t 2 48 10 "--no_count --save_result"
#########################################


# FSRCNN
model=$1
# patch=`expr $3 \* 48`
patch=`expr $3 \* $4`
if [ $2 = "xt" ]; then
  dim_configs="--repecn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24"
elif [ $2 = "t" ]; then
  dim_configs="--repecn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30"
elif [ $2 = "s" ]; then
  dim_configs="--repecn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60"
elif [ $2 = "b" ]; then
  dim_configs="--repecn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180"
elif [ $2 = "fblt" ]; then
  dim_configs="--repecn_up_feat 42 --depths 6+6+6 --dims 42+42+42"
elif [ $2 = "fbt" ]; then
  dim_configs="--repecn_up_feat 30 --depths 6+6+6 --dims 30+30+30"
elif [ $2 = "fbxt" ]; then
  dim_configs="--repecn_up_feat 24 --depths 6+6 --dims 24+24"
elif [ $2 = "fs" ]; then
  dim_configs="--repecn_up_feat 60 --depths 6+6+6+6+6 --dims 60+60+60+60+60"
elif [ $2 = "base" ]; then
  dim_configs="--ShMn_feats 64 --ShMkSize 7"
elif [ $2 = "tiny" ]; then
  dim_configs="--ShMn_feats 32 --ShMkSize 3"
elif [ $2 = "v1t" ]; then
  dim_configs="--repecn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4"
elif [ $2 = "v1s" ]; then
  dim_configs="--repecn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --mlp_ratios 4+4+4"
elif [ $2 = "v1b" ]; then
  dim_configs="--repecn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --mlp_ratios 4+4+4+4"
elif [ $2 = "b26" ]; then
  dim_configs="--repecn_up_feat 128 --depths 6+6+6+6+6+6 --dims 128+128+128+128+128+128 --mlp_ratios 4+4+4+4+4+4"
elif [ $2 = "t2" ]; then
  dim_configs="--repecn_up_feat 48 --depths 6+6+6 --dims 48+48+48 --mlp_ratios 4+4+4"
elif [ $2 = "xt2" ]; then
  dim_configs="--repecn_up_feat 32 --depths 4+4 --dims 32+32 --mlp_ratios 4+4"
fi

if [ $model = "FSRCNN" ]; then
  # trained by: python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --lr 1e-3 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --model FSRCNN --reset --save fsrcnn_x2
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model FSRCNN --save ../runtime_models/logs/fsrcnn_s64_x$3 --pre_train ../runtime_models/fsrcnn_s64_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "SRCNN" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model SRCNN --save ../runtime_models/logs/srcnn_s64_x$3 --pre_train ../runtime_models/srcnn_s64_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "v8" ]; then
  # ./scripts/train_repecn_v8.sh runtime 0 1 xt ab $3 48 ms skip v8old $5
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --res_connect skip $dim_configs --upsampling Nearest --acb_norm v8old --model RepECNV8 --save ../runtime_models/v8$2_x$3 --pre_train ../runtime_models/logs/v8$2_x$3.pt --test_only --load_inf --reset --runtime --times $5 $6 #--no_count --save_result 
# elif [ $model = "v5" ]; then
#   python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --res_connect 1acb3 $dim_configs --model RepECNV5 --save ../runtime_models/logs/v5$2_x$3 --pre_train ../runtime_models/v5$2_x$3.pt --test_only --load_inf --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "v5" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --res_connect 1acb3 $dim_configs --model RepECNV5 --save ../runtime_models/logs/v5$2_x$3 --pre_train ../runtime_models/v5$2_x$3.pt --test_only --load_inf --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "IMDN" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model IMDN --save ../runtime_models/logs/IMDN_x$3 --pre_train ../runtime_models/IMDN_x$3.pth --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "LAPAR_A" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model LAPAR_A --save ../runtime_models/logs/LAPAR_A_x$3 --pre_train ../runtime_models/LAPAR_A_x$3.pth --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "LAPAR_B" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model LAPAR_B --save ../runtime_models/logs/LAPAR_B_x$3 --pre_train ../runtime_models/LAPAR_B_x$3.pth --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "LAPAR_C" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model LAPAR_C --save ../runtime_models/logs/LAPAR_C_x$3 --pre_train ../runtime_models/LAPAR_C_x$3.pth --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "CARN" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model CARN --save ../runtime_models/logs/CARN_x$3 --pre_train ../runtime_models/CARN_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "LatticeNet" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model LatticeNet --save ../runtime_models/logs/LatticeNet_x$3 --pre_train ../runtime_models/LatticeNet_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "LBNet" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --n_feats 32 --num_heads 8 --model LBNet --save ../runtime_models/logs/LBNet_x$3 --pre_train ../runtime_models/LBNet-X$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "LBNet-T" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --n_feats 18 --num_heads 6 --model LBNet --save ../runtime_models/logs/LBNet-T_x$3 --pre_train ../runtime_models/LBNet-T_X$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "ESRT" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model ESRT --save ../runtime_models/logs/ESRT_x$3 --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "IDN" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model IDN --save ../runtime_models/logs/IDN_x$3 --pre_train ../runtime_models/IDN_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "LapSRN" ]; then
  echo "python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 1 --model LapSRN --save ../runtime_models/logs/LapSRN_x$3 --pre_train ../runtime_models/LapSRN_x$3.pt --test_only --reset --runtime --times $5 $6"
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 1 --model LapSRN --save ../runtime_models/logs/LapSRN_x$3 --pre_train ../runtime_models/LapSRN_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "DRRN" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 1 --model DRRN --save ../runtime_models/logs/DRRN_x$3 --pre_train ../runtime_models/DRRN_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "SwinIR" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --res_connect 1acb3 $dim_configs --model SwinIR --save ../runtime_models/logs/SwinIR-$2_x$3 --pre_train ../runtime_models/SwinIR-$2_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "EDSR-baseline" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model EDSR --save ../runtime_models/logs/EDSR-baseline_x$3 --pre_train ../models/EDSR-baseline_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "EDSR" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model EDSR --n_resblocks 32 --n_feats 256 --res_scale 0.1 --save ../runtime_models/logs/EDSR_x$3  --pre_train ../models/EDSR_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "RCAN" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --template RCAN --save ../runtime_models/logs/RCAN_x$3 --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "ShuffleMixer" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --data_test 720P --n_colors 3 $dim_configs --model ShuffleMixer --save ../runtime_models/logs/ShuffleMixer-$2_x$3 --pre_train ../runtime_models/shufflemixer_$2_x$3.pth --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "ALAN" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --data_test 720P --n_colors 3 $dim_configs --model RAAN --interpolation Nearest --acb_norm batch --upsampling Nearest --use_acb --save ../runtime_models/logs/alan_$2_x$3 --pre_train ../runtime_models/alan_$2_x$3.pt --test_only --load_inf --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "DCANV1" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --data_test 720P --n_colors 3 $dim_configs --model DCAN --interpolation Nearest --acb_norm no --stage_res --upsampling Nearest --bb_norm BN --degradation bicubic --sigma 0 --quality 0 --pre_train ../runtime_models/dcanv1_$2_x$3.pt --save ../runtime_models/logs/dcanv1_$2_x$3 --test_only --load_inf --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "DCANV2" ]; then
  python main.py --n_threads 6 --scale $3 --patch_size $patch --data_test 720P --n_colors 3 $dim_configs --model DCANV2 --interpolation Nearest --acb_norm no --stage_res --upsampling Nearest --bb_norm BN --degradation bicubic --sigma 0 --quality 0 --pre_train ../runtime_models/dcanv2_$2_x$3.pt --save ../runtime_models/logs/dcanv2_$2_x$3 --test_only --load_inf --reset --runtime --times $5 $6 #--no_count --save_result 
else
  # echo "The model $1 is not supported!"
  python main.py --n_threads 6 --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model $model --save ../runtime_models/logs/${model}_x$3 --pre_train ../runtime_models/${model}_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
fi
