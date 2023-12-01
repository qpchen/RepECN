import torch
import copy
import time
import os.path
from os import path as osp

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from torchinfo import summary
from thop import profile, clever_format
from tqdm import tqdm

from util.interpret_tool import get_model_interpretation

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def main():
    global model
    if checkpoint.ok:
        _model = model.Model(args, checkpoint)
        _input_path = args.dir_demo
        loader = data.Data(args)
        # _is_test = True#args.test_only
        # _device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu')
        # if args.save_results: checkpoint.begin_background()

        for idx_data, d in enumerate(loader.loader_test):
            for idx_scale, scale in enumerate(args.scale):
                for lr, hr, filename in tqdm(d, ncols=80):
                    _img_path = osp.join(_input_path, filename[0]) + '.png'
                    img, di = get_model_interpretation(_model, _img_path, idx_scale,# img_opt['w'], img_opt['h'],
                                           use_cuda=not args.cpu)
                    save_list = [img]
                    # if args.save_results: checkpoint.save_results(d, filename[0], save_list, scale)
                    os.makedirs(osp.join(args.save, 'interpretation'), exist_ok=True)
                    img.save(osp.join(args.save, 'interpretation', filename[0]+'.png'))
                    checkpoint.write_log(f"DI of {filename[0]}: {round(di, 3)}")
        # if args.save_results: checkpoint.end_background()
        
        checkpoint.done()


if __name__ == '__main__':
    main()


# ALAN: ablation 2. LKA使用不同核
# 7: python lam.py --n_GPUs 1 --save_result --data_test Demo --dir_demo ../lams/imgs --scale 4 --test_only --load_inf --patch_size 192 --n_colors 3 --model RAAN --interpolation Nearest --acb_norm batch --upsampling Nearest --use_acb --repecn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --DWDkSize 7 --DWDdil 1 --pre_train ../raan_UpNN_ACBbatch_noStgRes_AddNr_MS_4e-4/v1xt_p48_LK7_x2_test/model/inf_model.pt --save ../lams/alan-v1xt_LK7
# 15: python lam.py --n_GPUs 1 --save_result --data_test Demo --dir_demo ../lams/imgs --scale 4 --test_only --load_inf --patch_size 192 --n_colors 3 --model RAAN --interpolation Nearest --acb_norm batch --upsampling Nearest --use_acb --repecn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --DWDkSize 7 --DWDdil 2 --pre_train ../raan_UpNN_ACBbatch_noStgRes_AddNr_MS_4e-4/v1xt_p48_LK14_x2_test/model/inf_model.pt --save ../lams/alan-v1xt_LK15
# 23: python lam.py --n_GPUs 1 --save_result --data_test Demo --dir_demo ../lams/imgs --scale 4 --test_only --load_inf --patch_size 192 --n_colors 3 --model RAAN --interpolation Nearest --acb_norm batch --upsampling Nearest --use_acb --repecn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --DWDkSize 7 --DWDdil 3 --pre_train ../raan_UpNN_ACBbatch_noStgRes_AddNr_MS_4e-4/v1xt_p48_x2_test/model/inf_model.pt --save ../lams/alan-v1xt_LK23
# 31: python lam.py --n_GPUs 1 --save_result --data_test Demo --dir_demo ../lams/imgs --scale 4 --test_only --load_inf --patch_size 192 --n_colors 3 --model RAAN --interpolation Nearest --acb_norm batch --upsampling Nearest --use_acb --repecn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --DWDkSize 7 --DWDdil 4 --pre_train ../raan_UpNN_ACBbatch_noStgRes_AddNr_MS_4e-4/v1xt_p48_LK28_x2_test/model/inf_model.pt --save ../lams/alan-v1xt_LK31
# 39: python lam.py --n_GPUs 1 --save_result --data_test Demo --dir_demo ../lams/imgs --scale 4 --test_only --load_inf --patch_size 192 --n_colors 3 --model RAAN --interpolation Nearest --acb_norm batch --upsampling Nearest --use_acb --repecn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --DWDkSize 7 --DWDdil 5 --pre_train ../raan_UpNN_ACBbatch_noStgRes_AddNr_MS_4e-4/v1xt_p48_LK39_x2_test/model/inf_model.pt --save ../lams/alan-v1xt_LK39
# 47: python lam.py --n_GPUs 1 --save_result --data_test Demo --dir_demo ../lams/imgs --scale 4 --test_only --load_inf --patch_size 192 --n_colors 3 --model RAAN --interpolation Nearest --acb_norm batch --upsampling Nearest --use_acb --repecn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --DWDkSize 7 --DWDdil 6 --pre_train ../raan_UpNN_ACBbatch_noStgRes_AddNr_MS_4e-4/v1xt_p48_LK47_x2_test/model/inf_model.pt --save ../lams/alan-v1xt_LK47
# 55: python lam.py --n_GPUs 1 --save_result --data_test Demo --dir_demo ../lams/imgs --scale 4 --test_only --load_inf --patch_size 192 --n_colors 3 --model RAAN --interpolation Nearest --acb_norm batch --upsampling Nearest --use_acb --repecn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --DWDkSize 7 --DWDdil 7 --pre_train ../raan_UpNN_ACBbatch_noStgRes_AddNr_MS_4e-4/v1xt_p48_LK55_x2_test/model/inf_model.pt --save ../lams/alan-v1xt_LK55

# SwinIR (seems LAM cannot applied for scale 2, only for scale 4)
# original x4: python lam.py --n_GPUs 1 --scale 4 --patch_size 256 --n_colors 3 --repecn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --res_connect 1conv1 --data_test Demo --dir_demo ../lams/imgs --pre_train ../experiment/SwinIR_t_x4/model/model_best.pt --model SwinIR --save ../lams/SwinIR_t_x4_lam --test_only --save_result
# original x2(location error): python lam.py --n_GPUs 1 --scale 2 --patch_size 128 --n_colors 3 --repecn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --res_connect 1conv1 --data_test Demo --dir_demo ../lams/imgs --pre_train ../experiment/SwinIR_t/model/model_best.pt --model SwinIR --save ../lams/SwinIR_t_x2_lam --test_only --save_result
# dwd: python lam.py --n_GPUs 1 --scale 4 --patch_size 256 --n_colors 3 --repecn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --res_connect 1conv1 --data_test Demo --dir_demo ../lams/imgs --pre_train ../experiment/SwinIRA_t_x4/model/model_best.pt --model SwinIRA --swinir_ablation --save ../lams/SwinIRA_t_x4_lam --test_only --save_result
# dwd+ffn: python lam.py --n_GPUs 1 --scale 4 --patch_size 256 --n_colors 3 --repecn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --res_connect 1conv1 --data_test Demo --dir_demo ../lams/imgs --pre_train ../experiment/SwinIRA_t_DWMlp_x4/model/model_best.pt --model SwinIRA --swinir_ablation --swinir_mlpdwc --save ../lams/SwinIRA_t_DWMlp_x4_lam --test_only --save_result