import torch
import copy

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
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            if args.model == 'UFSRCNN' and not args.test_only:
                downargs = copy.deepcopy(args)
                downargs.model = 'DFSRCNN'
                downargs.save = 'dfsrcnn_v1_x2'
                downargs.load = ''
                downargs.resume = 0
                downargs.reset = False
                downckp = utility.checkpoint(downargs)
                downmodel = model.Model(downargs, downckp)
            elif (args.model == 'UFSRCNNPS' or args.model == 'UFSRCNNPSV2' \
                    or args.model == 'UFSRCNNPSV6'  or args.model == 'UFSRCNNPSV7') \
                    and not args.test_only:
                downargs = copy.deepcopy(args)
                downargs.model = 'DFSRCNNPS'
                if args.scale[0] == 2:
                    downargs.save = 'dfsrcnnps_v1_x2'
                elif args.scale[0] == 3:
                    downargs.save = 'dfsrcnnps_v1_x3'
                elif args.scale[0] == 4:
                    downargs.save = 'dfsrcnnps_v1_x4'
                downargs.load = ''
                downargs.resume = 0
                downargs.reset = False
                downckp = utility.checkpoint(downargs)
                downmodel = model.Model(downargs, downckp)
            else:
                downmodel = None
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            if not args.no_count:
                ################# get model info manually #####################
                # param = utility.count_param(_model, args.model)
                # print('%s total parameters: %.2fM (%d)' % (args.model, param / 1e6, param))
                ################# get model info from torchinfo #####################
                _device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu')
                _is_test = True#args.test_only
                _batch_size = args.batch_size if not _is_test else 1
                _mode = "train" if not _is_test else "eval"
                _height = args.patch_size if not _is_test else 1280 // args.scale[0]#args.patch_size#1280
                _width =  args.patch_size if not _is_test else 720 // args.scale[0]##args.patch_size#720
                _input_size = (_batch_size, args.n_colors, _height, _width)
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
                    _input = torch.randn(_input_size)
                    _input = _input.to(_device)
                    _macs, _params = profile(_model, inputs=(_input, args.scale))
                    _m_fm, _p_fm = clever_format([_macs, _params], "%.3f")#inputs=(_input, args.scale))
                    _trop = str('\nThop: %s\'s total Parameters: %s(%d); MACs(Multi-Adds): %s(%d).\n' % (args.model, _p_fm, _params, _m_fm, _macs))
                    # print(_trop)
                    checkpoint.write_log(_trop)
                    _input = _input.to(torch.device('cpu'))
                if torch.cuda.is_available(): torch.cuda.empty_cache()  # release the cache costs on GPU by the statistics function
            t = Trainer(args, loader, _model, _loss, checkpoint, downmodel)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()


if __name__ == '__main__':
    main()
