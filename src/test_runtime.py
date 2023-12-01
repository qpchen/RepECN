import torch
import copy
import time

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from torchinfo import summary
from thop import profile, clever_format

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def main():
    global model
    if checkpoint.ok:
        _model = model.Model(args, checkpoint)
        _is_test = True#args.test_only
        _batch_size = 1
        _height = 1280 // args.scale[0]#args.patch_size#1280
        _width =  720 // args.scale[0]##args.patch_size#720
        _input_size = (_batch_size, args.n_colors, _height, _width)
        _input = torch.randn(_input_size)
        _device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu')
        _input = _input.to(_device)
        if not args.no_count:
            ################# get model info manually #####################
            # param = utility.count_param(_model, args.model)
            # print('%s total parameters: %.2fM (%d)' % (args.model, param / 1e6, param))
            ################# get model info from torchinfo #####################
            _mode = "train" if not _is_test else "eval"
            checkpoint.write_log(str(
                summary(_model, 
                        input_size = (_input_size, args.scale),
                        col_names = ("output_size", "num_params", "params_percent", "mult_adds"),
                        depth = 6, 
                        mode = _mode, 
                        row_settings = ("depth", "var_names"),
                        verbose=0)))
            # checkpoint.write_log(str(summary(_model, [(_batch_size, args.n_colors, 1280, 720), args.scale])))
            ################# get model info from thop #####################
            if args.n_GPUs == 1:  # TODO: fix the thop error when use data_parallel in training. some modules are on device cpu
                _macs, _params = profile(_model, inputs=(_input, args.scale))
                _m_fm, _p_fm = clever_format([_macs, _params], "%.3f")#inputs=(_input, args.scale))
                _trop = str('\nThop: %s\'s total Parameters: %s(%d); MACs(Multi-Adds): %s(%d).\n' % (args.model, _p_fm, _params, _m_fm, _macs))
                # print(_trop)
                checkpoint.write_log(_trop)
                # _input = _input.to(torch.device('cpu'))
            if torch.cuda.is_available(): torch.cuda.empty_cache()  # release the cache costs on GPU by the statistics function
        if args.runtime:
            sum_run = 0
            # sum_run_2 = 0
            _times = args.times
            for _i in range(_times):
                run_start = time.time()
                sr = _model(_input, 0)
                run_stop = time.time()
                runtime = run_stop - run_start
                checkpoint.write_log('No.{} Runtime: {:.2f}ms.'.format(_i + 1, runtime*1e3))
                sum_run += runtime
                # if _i != 0:
                #     sum_run_2 += runtime
            checkpoint.write_log('Average Runtime: {:.2f}ms.\n'.format((sum_run/_times)*1e3))

        checkpoint.done()


if __name__ == '__main__':
    main()
