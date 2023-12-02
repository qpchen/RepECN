#!/bin/bash

################################################################################
######################      RepECN V9_D1acb3 noACBnorm befln nolr 2e-4 bicubic 0 0       ######################
################################################################################
# ./scripts/train_repecn.sh [mode] [cuda_device] [accummulation_step] [model] [interpolation] [sr_scale] [lr_patch_size] [LR_scheduler_class] [init LR] [block_conv] [deep_conv] [acb_norm] [LN] [addLR]
# run example repecn_test_D1acb3_x2: ./scripts/train_repecn.sh train 0 1 test b 2 48 ms skip 1acb3 batch befln nolr 2e-4 bicubic 0 0
# ########### training commands ###########

# batch_befLN/repecn_t_s64_1acb3_D1acb3_x2: ./scripts/train_repecn.sh train 0 1 t b 2 64 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0
# batch_befLN/repecn_t_s64_1acb3_D1acb3_x3: ./scripts/train_repecn.sh train 1 1 t b 3 64 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0
# batch_befLN/repecn_t_s64_1acb3_D1acb3_x4: ./scripts/train_repecn.sh train 2 1 t b 4 64 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0

# batch_befLN/repecn_s_s64_1acb3_D1acb3_x2: ./scripts/train_repecn.sh train 1 1 s b 2 64 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0
# batch_befLN/repecn_s_s64_1acb3_D1acb3_x3: ./scripts/train_repecn.sh train 2 1 s b 3 64 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0
# batch_befLN/repecn_s_s64_1acb3_D1acb3_x4: ./scripts/train_repecn.sh train 3 1 s b 4 64 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0

# batch_befLN/repecn_b_1acb3_D1acb3_x2: ./scripts/train_repecn.sh train 1 1 b b 2 48 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0
# batch_befLN/repecn_b_1acb3_D1acb3_x3: ./scripts/train_repecn.sh train 2 1 b b 3 48 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0
# batch_befLN/repecn_b_1acb3_D1acb3_x4: ./scripts/train_repecn.sh train 3 1 b b 4 48 ms 1acb3 1acb3 batch befln nolr 2e-4 bicubic 0 0


# #####################################
# accept input
# input 2 params, first is run mode, 
mode=$1
# second is devices of gpu to use
device=$2
n_device=`expr ${#device} / 2 + 1`
# third is accumulation_step number
accum=$3
# forth is model size
size=$4

# ############## model_base #############
if [ $size = "b" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --repecn_up_feat 60 --depths 6+6+6+6+6 --dims 60+60+60+60+60 --batch_size 32"
# ############## model_small #############
elif [ $size = "s" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --upsampling Nearest --repecn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --batch_size 32"
# ############## model_tiny #############
elif [ $size = "t" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --upsampling Nearest --repecn_up_feat 24 --depths 6+6 --dims 24+24 --batch_size 32"
# ############## test_model #############
elif [ $size = "test" ]; then  # test with lower costs
  options="--epochs 3000 --decay 1500-2400-2700-2850 --upsampling Nearest --repecn_up_feat 6 --depths 2+4 --dims 6+12 --batch_size 4"
else
  echo "no this size $size !"
  exit
fi
# if the output add interpolation of input
interpolation=$5
if [ $interpolation = "b" ]; then
  interpolation_print=""
  interpolation=""
elif [ $interpolation = "bl" ]; then
  interpolation_print="_Biln"
  interpolation="--interpolation Bilinear"
elif [ $interpolation = "n" ]; then
  interpolation_print="_Nrst"
  interpolation="--interpolation Nearest"
elif [ $interpolation = "s" ]; then
  interpolation_print="_Skip"
  interpolation="--interpolation Skip"
elif [ $interpolation = "p" ]; then
  interpolation_print="_PxSh"
  interpolation="--interpolation PixelShuffle"
else
  echo "no valid $interpolation ! Please input (b | bl | n | p | s)."
fi
# fifth is sr scale
scale=$6
# sixth is the LQ image patch size
patch=$7
patch_hr=`expr $patch \* $scale`
if [ $patch = 48 ]; then
  patch_print=""
else
  patch_print="_s$patch"
fi

# lr_class choice, default is MultiStepLR. test whether CosineWarmRestart can be better
# if [ $# == 8 ]; then
  lr=$8
  if [ $lr = "cosre" ]; then  # CosineWarmRestart
    lr_class="CosineWarmRestart"
    lr_print="_CR"
  elif [ $lr = "cos" ]; then  # CosineWarm
    lr_class="CosineWarm"
    lr_print="_C"
  else  # $lr = "ms"
    lr_class="MultiStepLR"
    lr_print=""
  fi
# else
#   lr_class="MultiStepLR"
#   lr_print=""
# fi
# res_connect choice, other version default is 1acb3(same as version 5). 
# But repecn_ default 'skip'. other choices are 1conv1 3acb3
res=$9
if [ $res = "skip" ]; then
  res_print=""
else
  res_print="_$res"
fi
deep=${10}
# the last conv at end of deep feature module, before skip connect
if [ $deep = "skip" ]; then
  deep_print=""
else
  deep_print="_D$deep"
fi
# acb norm choices, can be "batch", "inst", "no", "v8old"
acb=${11}
acb_print="_$acb"
# backbone norm choices
norm=${12}
if [ $norm = "ln" ]; then
  norm_opt="--norm_at after"
  norm_print=""
elif [ $norm = "no" ]; then  # "no": means do not use norm
  norm_opt="--no_layernorm"
  norm_print="_noLN"
elif [ $norm = "befln" ]; then
  norm_opt="--norm_at before"
  norm_print="_befLN"
fi
# add lr to upsampling, can be "addlr" or "nolr"
addlr=${13}
if [ $addlr = "addlr" ]; then
  addlr_opt="--add_lr"
  addlr_print="_addlr"
elif [ $addlr = "nolr" ]; then
  addlr_opt=""
  addlr_print=""
fi
initlr=${14}
if [ $initlr = "2e-4" ]; then
  initlr_print=""
else
  initlr_print="_$initlr"
fi
# degradation option bicubic, BD, DN, Noise, Blur, JPEG, Gray_Noise
deg=${15}
deg_opt="--degradation $deg"
deg_print="_$deg"
if [ $deg = "bicubic" ]; then
  deg_print=""
  val_set="Set5"
  test_set="Set5+Set14+B100+Urban100+Manga109"
elif [ $deg = "BD" ]; then 
  val_set="Set5"
  test_set="Set5+Set14+B100+Urban100+Manga109"
elif [ $deg = "DN" ]; then 
  val_set="Set5"
  test_set="Set5+Set14+B100+Urban100+Manga109"
elif [ $deg = "Noise" ]; then 
  val_set="McMaster"
  test_set="McMaster+Kodak24+CBSD68+Urban100"
elif [ $deg = "Gray_Noise" ]; then 
  deg_opt="$deg_opt --n_colors 1"
  val_set="Set12"
  test_set="Set12+BSD68+Urban100_Gray"
elif [ $deg = "Blur" ]; then 
  val_set="McMaster"
  test_set="McMaster+Kodak24+Urban100"
elif [ $deg = "JPEG" ]; then 
  val_set="Classic5"
  test_set="Classic5+LIVE1"
fi
# number of sigma for Noise degradation (15 | 25 | 50)
sigma=${16}
if [ $sigma = 0 ]; then
  sigma_print=""
else
  sigma_print="_N$sigma"
fi
# number of quality for JPEG degradation (10 | 20 | 30 | 40)
quality=${17}
if [ $quality = 0 ]; then
  quality_print=""
else
  quality_print="_Q$quality"
fi


# #####################################
# prepare program options parameters
# repecn_ must use layernorm
run_command="--n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options $interpolation --res_connect $res --deep_conv $deep --acb_norm $acb --loss 1*SmoothL1 --lr $initlr --optimizer ADAM --skip_threshold 1e6 --lr_class $lr_class $norm_opt $addlr_opt --data_train DIV2K_IR --data_test $val_set $deg_opt --sigma $sigma --quality $quality --model RepECN"
# run_command="--n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options $interpolation --res_connect $res --deep_conv $deep --loss 1*L1 --lr $initlr --optimizer ADAM --skip_threshold 1e6 --lr_class CosineWarmRestart --model RepECN"
save_dir="../repecn${acb_print}${norm_print}${initlr_print}/repecn_${size}${patch_print}${addlr_print}${interpolation_print}${res_print}${deep_print}${lr_print}${deg_print}${sigma_print}${quality_print}_x${scale}"
log_file="../repecn${acb_print}${norm_print}${initlr_print}/logs/repecn_${size}${patch_print}${addlr_print}${interpolation_print}${res_print}${deep_print}${lr_print}${deg_print}${sigma_print}${quality_print}_x${scale}.log"

if [ ! -d "../repecn${acb_print}${norm_print}${initlr_print}" ]; then
  mkdir "../repecn${acb_print}${norm_print}${initlr_print}"
fi
if [ ! -d "../repecn${acb_print}${norm_print}${initlr_print}/logs" ]; then
  mkdir "../repecn${acb_print}${norm_print}${initlr_print}/logs"
fi


# #####################################
# run train/eval program
export CUDA_VISIBLE_DEVICES=$device
echo "CUDA GPUs use: No.'$CUDA_VISIBLE_DEVICES' devices."

if [ $mode = "train" ]
then
  if [ -f "$save_dir/model/model_latest.pt" ]; then
    echo "$save_dir seems storing some model files trained before, please change the save dir!"
  else
    echo "start training from the beginning:"
    echo "nohup python main.py $run_command --save $save_dir --reset > $log_file 2>&1 &"
    nohup python main.py $run_command --save $save_dir --reset > $log_file 2>&1 &
  fi
elif [ $mode = "resume" ]
then
  echo "resume training:"
  echo "nohup python main.py $run_command --load $save_dir --resume -1 > $log_file 2>&1 &"
  nohup python main.py $run_command --load $save_dir --resume -1 >> $log_file 2>&1 &
elif [ $mode = "switch" ]
then
  echo "switch acb from training to inference mode:"
  echo "python main.py $run_command --save ${save_dir}_test --pre_train $save_dir/model/model_best.pt --test_only --inf_switch"
  python main.py $run_command --save ${save_dir}_test --pre_train $save_dir/model/model_best.pt --test_only --inf_switch
elif [ $mode = "eval" ]
then
  echo "load inference version of acb to eval:"
  echo "python main.py $run_command --data_test $test_set --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf"
  python main.py $run_command --data_test $test_set --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf
elif [ $mode = "eval_plus" ]
then
  echo "load inference version of acb to eval:"
  echo "python main.py $run_command --data_test $test_set --save ${save_dir}_test_plus --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf --self_ensemble"
  python main.py $run_command --data_test $test_set --save ${save_dir}_test_plus --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf --self_ensemble
elif [ $mode = "runtime" ]
then
  # echo "load inference version of acb to test the runtime:"
  # echo "python main.py $run_command --data_test 720P --runtime --no_count --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf"
  python main.py $run_command --data_test 720P --runtime --no_count --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf --times ${11}
elif [ $mode = "lam" ]
then
  echo "doing Local attribution map (LAM) analysis:"
  echo "python lam.py $run_command --data_test Demo --dir_demo ../lams/imgs --save ${save_dir}_lam --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf"
  python lam.py $run_command --data_test Demo --dir_demo ../lams/imgs --save ${save_dir}_lam --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf
else
  echo "invalid value, it only accpet train, resume, switch, eval!"
fi

