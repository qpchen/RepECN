#!/bin/bash

################################################################################
######################      running command       ######################
################################################################################
# ./scripts/train_alan.sh [mode] [cuda_device] [accummulation_step] [model_size] [interpolation] [sr_scale] [lr_patch_size] [LR_scheduler_class] [init LR] [stage Res] [acb_norm] [upsampling]

# ############### ALAN #################
# saved files are under dir: ALANV2_UpNN_ACBno_StgRes_AddNr_MS_5e-4/
# ./scripts/train_alan.sh train 1 1 t nr 4 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no SmoothL1
# ./scripts/train_alan.sh switch 0 1 t nr 3 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no SmoothL1
# ./scripts/train_alan.sh eval 1 1 t nr 2 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no SmoothL1

# ./scripts/train_alan.sh train 0 1 s nr 2 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no SmoothL1
# ./scripts/train_alan.sh switch 1 1 s nr 3 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no SmoothL1
# ./scripts/train_alan.sh eval 1 1 s nr 4 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no SmoothL1

# ./scripts/train_alan.sh train 0 1 b nr 2 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no SmoothL1
# ./scripts/train_alan.sh switch 1 1 b nr 3 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no SmoothL1
# ./scripts/train_alan.sh eval 1 1 b nr 4 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no SmoothL1

# ############### ALAN ablation #################
# 2. LKA size; 3. MLP ratio；4. LKA不使用attention; 5. stage、block等数量设置；6. Loss; 7. no PA
# 2: ./scripts/train_alan.sh train 0 1 s nr 2 48 ms 5e-4 useStageRes no NN ACB 7 BN bicubic 0 0 no SmoothL1
# 2: ./scripts/train_alan.sh train 0 1 s nr 2 48 ms 5e-4 useStageRes no NN ACB 15 BN bicubic 0 0 no SmoothL1
# 2: ./scripts/train_alan.sh train 0 1 s nr 2 48 ms 5e-4 useStageRes no NN ACB 31 BN bicubic 0 0 no SmoothL1
# 2: ./scripts/train_alan.sh train 0 1 s nr 2 48 ms 5e-4 useStageRes no NN ACB 39 BN bicubic 0 0 no SmoothL1
# 2: ./scripts/train_alan.sh train 0 1 s nr 2 48 ms 5e-4 useStageRes no NN ACB 47 BN bicubic 0 0 no SmoothL1
# 2: ./scripts/train_alan.sh train 0 1 s nr 2 48 ms 5e-4 useStageRes no NN ACB 55 BN bicubic 0 0 no SmoothL1
# 3: ./scripts/train_alan.sh train 0 1 s_mlp2 nr 2 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no SmoothL1
# 4: ./scripts/train_alan.sh train 1 1 s_noAttn nr 2 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no SmoothL1
# 5: ./scripts/train_alan.sh train 0 1 s nr 2 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no SmoothL1
# 6: ./scripts/train_alan.sh train 1 1 s nr 2 48 ms 5e-4 useStageRes no NN ACB 23 BN bicubic 0 0 no L1
# 7: ./scripts/train_alan.sh train 0 1 s nr 2 48 ms 5e-4 useStageRes no NNnPA ACB 23 BN bicubic 0 0 no SmoothL1




# #####################################
# accept input
# first is run mode, 
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
  options="--epochs 1500 --decay 750-1200-1350-1425 --repecn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --mlp_ratios 4+4+4+4 --batch_size 32"
# ############## model_small #############
elif [ $size = "s" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --repecn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --mlp_ratios 4+4+4 --batch_size 32"
elif [ $size = "s_mlp2" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --repecn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --mlp_ratios 2+2+2 --batch_size 32"
elif [ $size = "s_noAttn" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --repecn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --mlp_ratios 4+4+4 --batch_size 32 --no_attn"
# ############## model_tiny #############
elif [ $size = "t" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --repecn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --batch_size 32"
elif [ $size = "t_mlp2" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --repecn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 2+2 --batch_size 32"
elif [ $size = "t_noAttn" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --repecn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --batch_size 32 --no_attn"
# ############## test_model #############
elif [ $size = "test" ]; then  # test with lower costs
  options="--epochs 3000 --decay 1500-2400-2700-2850 --repecn_up_feat 6 --depths 2+4 --dims 6+12 --mlp_ratios 4+4 --batch_size 4"
else
  echo "no this size $size !"
  exit
fi
# if the output add interpolation of input
interpolation=$5
if [ $interpolation = "bc" ]; then
  interpolation_print=""
  interpolation=""
elif [ $interpolation = "bl" ]; then
  interpolation_print="_AddBL"
  interpolation="--interpolation Bilinear"
elif [ $interpolation = "nr" ]; then
  interpolation_print="_AddNr"
  interpolation="--interpolation Nearest"
elif [ $interpolation = "sk" ]; then
  interpolation_print="_AddSk"
  interpolation="--interpolation Skip"
elif [ $interpolation = "ps" ]; then
  interpolation_print="_AddPS"
  interpolation="--interpolation PixelShuffle"
else
  echo "no valid $interpolation ! Please input (bc | bl | nr | ps | sk)."
fi
# fifth is sr scale
scale=$6
# sixth is the LQ image patch size
patch=$7
patch_hr=`expr $patch \* $scale`
patch_print="_p$patch"
# lr_class choice, default is MultiStepLR. test whether CosineWarmRestart can be better
lr=$8
if [ $lr = "cosre" ]; then  # for CosineWarmRestart
  lr_class="CosineWarmRestart"
  lr_print="_CWRe"
elif [ $lr = "cos" ]; then  # for CosineWarm
  lr_class="CosineWarm"
  lr_print="_CW"
else  # $lr = "ms"
  lr_class="MultiStepLR"
  lr_print="_MS"
fi
initlr=$9
if [ $initlr = "2e-4" ]; then
  initlr_print=""
else
  initlr_print="_$initlr"
fi
# stage level residual connect
stageres=${10}
if [ $stageres = "useStageRes" ]; then
  stageres_opt="--stage_res"
  stageres_print="_StgRes"
elif [ $stageres = "noStageRes" ]; then  # better on test! better on other?
  stageres_opt=""
  stageres_print="_noStgRes"
else
  echo "no valid $stageres ! Please input (useStageRes | noStageRes)."
  exit
fi
# acb norm choices, can be "batch", "inst", "no", "v8old"
acb=${11}
acb_print="_ACB$acb"
# upsampling optionsNearestNoPA
upsam=${12}
if [ $upsam = "NN" ]; then  # best? use Nearest-Neibor
  upsam_print="_UpNN"
  upsam_opt="Nearest"
elif [ $upsam = "PSnA" ]; then  # nA better? use PixelShuffle with no activate layer, same as SwinIR
  upsam_print="_UpPSnA"
  upsam_opt="PixelShuffle --no_act_ps"
elif [ $upsam = "PS" ]; then  # worst? use PixelShuffle with activate layer
  upsam_print="_UpPS"
  upsam_opt="PixelShuffle"
elif [ $upsam = "NNnPA" ]; then  # worse? use Nearest-Neibor without pixel attention
  upsam_print="_UpNNnPA"
  upsam_opt="NearestNoPA"
else
  echo "no valid $upsam ! Please input (NN | PS | PSnA | NNnPA)."
  exit
fi
# use ACB or not
use_acb=${13}
if [ $use_acb = "ACB" ]; then  # best? use Nearest-Neibor
  use_acb_print=""
  use_acb_opt="--use_acb"
elif [ $use_acb = "noACB" ]; then 
  use_acb_print="_noACB"
  use_acb_opt=""
elif [ $use_acb = "DBB" ]; then 
  use_acb_print="_DBB"
  use_acb_opt="--use_acb --use_dbb"
else
  echo "no valid $use_acb ! Please input (ACB | DBB | noACB)."
  exit
fi
# set large kernel (LKA) size
LKAk=${14}
if [ $LKAk = "23" ]; then  # default
  LKAk_print=""
  LKAk_opt=""
elif [ $LKAk = "7" ]; then 
  LKAk_print="_LK7"
  LKAk_opt="--DWDkSize 7 --DWDdil 1"
elif [ $LKAk = "15" ]; then 
  LKAk_print="_LK15"
  LKAk_opt="--DWDkSize 7 --DWDdil 2"
elif [ $LKAk = "31" ]; then 
  LKAk_print="_LK31"
  LKAk_opt="--DWDkSize 7 --DWDdil 4"
elif [ $LKAk = "39" ]; then 
  LKAk_print="_LK39"
  LKAk_opt="--DWDkSize 7 --DWDdil 5"
elif [ $LKAk = "47" ]; then 
  LKAk_print="_LK47"
  LKAk_opt="--DWDkSize 7 --DWDdil 6"
elif [ $LKAk = "55" ]; then 
  LKAk_print="_LK55"
  LKAk_opt="--DWDkSize 7 --DWDdil 7"
else
  echo "no valid $LKAk ! Please input (7 | 15 | 23 | 31 | 39 | 47 | 55)."
  exit
fi
# backbone norm use BN | LN | no, without '--no_layernorm' to keep LN in each stage
bb_norm=${15}
# bb_norm_opt="--bb_norm $bb_norm"
if [ $bb_norm = "BN" ]; then  # best? use Nearest-Neibor
  bb_norm_opt="--bb_norm BN"
  bb_norm_print=""
elif [ $bb_norm = "LN" ]; then 
  bb_norm_opt="--bb_norm LN"
  bb_norm_print="_bbLN"
elif [ $bb_norm = "no" ]; then 
  bb_norm_opt="--bb_norm no"
  bb_norm_print="_bbnoN"
elif [ $bb_norm = "noAll" ]; then 
  bb_norm_opt="--bb_norm no --no_layernorm"
  bb_norm_print="_allnoN"
else
  echo "no valid $bb_norm ! Please input (BN | LN | no | noAll)."
  exit
fi
# degradation option bicubic, BD, DN, Noise, Blur, JPEG, Gray_Noise
deg=${16}
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
sigma=${17}
if [ $sigma = 0 ]; then
  sigma_print=""
else
  sigma_print="_N$sigma"
fi
# number of quality for JPEG  (10 | 20 | 30 | 40)
quality=${18}
if [ $quality = 0 ]; then
  quality_print=""
else
  quality_print="_Q$quality"
fi
# use feature down before extraction, and up after (feaDU | no)
feaDown=${19}
if [ $feaDown = "feaDU" ]; then  # old code down first
  feaDown_opt="--down_fea"
  feaDown_print="_feaDU"
elif [ $feaDown = "no" ]; then 
  feaDown_opt=""
  feaDown_print=""
elif [ $feaDown = "feaUD" ]; then
  feaDown_opt="--down_fea"
  feaDown_print="_feaUD"
else
  echo "no valid $feaDown ! Please input (feaDU | feaUD | no)."
  exit
fi
# loss function 
loss=${20}
if [ $loss = "SmoothL1" ]; then
  loss_print=""
else
  loss_print="_$loss"
fi


# #####################################
# prepare program options parameters
# v9 must use layernorm
run_command="--n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options $interpolation --acb_norm $acb $stageres_opt --upsampling $upsam_opt --loss 1*$loss --lr $initlr --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --lr_class $lr_class $use_acb_opt $LKAk_opt $bb_norm_opt $feaDown_opt --data_train DIV2K_IR --data_test $val_set $deg_opt --sigma $sigma --quality $quality --model ALAN"
father_dir="../ALAN${upsam_print}${use_acb_print}${acb_print}${stageres_print}${interpolation_print}${lr_print}${initlr_print}"
file_name="v1${size}${patch_print}${LKAk_print}${bb_norm_print}${feaDown_print}${deg_print}${sigma_print}${quality_print}${loss_print}_x${scale}"
save_dir="${father_dir}/${file_name}"
log_file="${father_dir}/logs/${file_name}.log"

if [ ! -d "${father_dir}" ]; then
  mkdir "${father_dir}"
fi
if [ ! -d "${father_dir}/logs" ]; then
  mkdir "${father_dir}/logs"
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

