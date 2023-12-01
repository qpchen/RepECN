import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,  # max use 4 for Pi system
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../../dataset',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-900',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--degradation', type=str, default='bicubic',
                    choices=('bicubic', 'BD', 'DN', 'Noise', 'Blur', 'JPEG', 'Gray_Noise'),
                    help='The degradation model for data (bicubic, BD, DN, Noise, Blur, JPEG, Gray_Noise)')
parser.add_argument('--sigma', type=int, default=15, 
                    help='number of sigma for Noise degradation (15 | 25 | 50)')
parser.add_argument('--quality', type=int, default=10, 
                    help='number of quality for JPEG degradation (10 | 20 | 30 | 40)')

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--num_heads', type=int, default=8,
                    help='The number of multi-head self-attention heads in the transformer')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')
parser.add_argument('--accumulation_step', type=int, default=1,
                    help='gradient accumulation step setting for larger batch')

# Optimization specifications
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate')
parser.add_argument('--lr_class', default='MultiStepLR',
                    choices=('MultiStepLR', 'CosineWarm', 'CosineWarmRestart'),
                    help='learning rate decay function (MultiStepLR | CosineWarm)')
parser.add_argument('--decay', type=str, default='250-400-450-475',
                    help='learning rate decay type for MultiStepLR milestone')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
# parser.add_argument('--T_max', type=int, default=3000,
#                     help='Maximum number of iterations from for CosineWarm, usually use epochs')
parser.add_argument('--T_0', type=int, default=5,
                    help='Number of iterations for the first restart of SGDR')
parser.add_argument('--T_mult', type=int, default=2,
                    help='A factor increases Ti after a restart of SGDR')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop', 'AdamW'),
                    help='optimizer to use (SGD | ADAM | RMSprop | AdamW)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')
parser.add_argument('--warmup', action='store_true',
                    help='Use warm up at the beginning of training')
parser.add_argument('--last_warm', type=int, default=20,
                    help='A factor of the last epoch of warm up, also used as period here')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

# GhostNetV2 reduce factor settings
parser.add_argument('--rf1', type=int, default='2',
                    help='GhostNetV2 reduce factor 1 stage')
parser.add_argument('--rf2', type=int, default='4',
                    help='GhostNetV2 reduce factor 2 stage')

# ACBlock or DBBlock inference-time settings
parser.add_argument('--use_acb', action='store_true',
                    help='use ACBlock instead of Conv2D')
parser.add_argument('--use_dbb', action='store_true',
                    help='use DBBlock instead of Conv2D, when use_acb set to true')
parser.add_argument('--load_inf', action='store_true',
                    help='Load ACBlock or DBBlock using inference-time Structure')
parser.add_argument('--inf_switch', action='store_true',
                    help='When test_only, load training-time Block, then switch to inference-time model and save it')
parser.add_argument('--acb_norm', default='batch',
                    choices=('batch', 'no', 'v8old', 'inst'),
                    help='last conv to use before residual connect (batch | no | v8old | inst)')

# SRACN settings
parser.add_argument('--n_feat', type=int, default=56,
                    help='number of channels in feature extraction layers')
parser.add_argument('--n_map_feat', type=int, default=12,
                    help='number of channels in mapping layers')
parser.add_argument('--n_up_feat', type=int, default=24,
                    help='number of channels in upsampling layers')

# RepECN deep extraction module settings
parser.add_argument('--depths', type=str, default='3+3',
                    help='Number of blocks at each stage')
parser.add_argument('--dims', type=str, default='12+24',
                    help='Feature dimension at each stage')
parser.add_argument('--repecn_up_feat', type=int, default=0,
                    help='number of feature maps in upsampling layers (default means same as dims[0])')
parser.add_argument('--drop_path_rate', type=float, default=0.,
                    help='Stochastic depth rate')
parser.add_argument('--layer_init_scale', type=float, default=1e-6,
                    help='Init value for Layer Scale')
parser.add_argument('--res_connect', default='1acb3',
                    choices=('1acb3', '3acb3', '1conv1', 'skip'),
                    help='last conv to use before residual connect (1acb3 | 3acb3 | 1conv1 | skip)')
parser.add_argument('--deep_conv', default='1acb3',
                    choices=('1acb3', '3acb3', '1conv1', 'skip'),
                    help='last conv to use at end of deep featurn module (1acb3 | 3acb3 | 1conv1 | skip)')

# RepECN UpSampling Function setting
parser.add_argument('--upsampling', default='Nearest',
                    choices=('Nearest', 'Deconv', 'PixelShuffle', 'PixelShuffleDirect', 'NearestNoPA'),
                    help='Upsampling to use (Nearest | Deconv | PixelShuffle | PixelShuffleDirect | NearestNoPA)')
parser.add_argument('--no_act_ps', action='store_true',
                    help='do not use activate function GELU in PixelShuffle')
                    
parser.add_argument('--no_bicubic', action='store_true',
                    help='do not add bicubic interpolation of input to output')
parser.add_argument('--add_lr', action='store_true',
                    help='do not add lr direct to upsampling after preup')
parser.add_argument('--interpolation', default='Bicubic',
                    choices=('Nearest', 'Bilinear', 'Bicubic', 'PixelShuffle', 'Skip'),
                    help='LR interpolation then add to output (Nearest | Bilinear | Bicubic | PixelShuffle | Skip')
parser.add_argument('--no_layernorm', action='store_true',
                    help='delete layer normalization for each acl/racb')
parser.add_argument('--norm_at', default='before',
                    choices=('after', 'before'),
                    help='layer norm location (after | before) the head_conv & connection of the resblocks')

parser.add_argument('--no_count', action='store_true',
                    help='Do model params and macs statistics')
parser.add_argument('--runtime', action='store_true',
                    help='print the runtime of model for each input')
parser.add_argument('--times', type=int, default=10,
                    help='running model for multi times to calculate the average runtime.')

# ALAN settings
parser.add_argument('--mlp_ratios', type=str, default='4+4',
                    help='MLp ratios at each stage')
parser.add_argument('--stage_res', action='store_true',
                    help='use residual connect at stage level')
# parser.add_argument('--LKAkSize', type=int, default=21,
#                     help='kernel size of convolution decomposed by LKA. (default 21, DW-D-Conv kernel=LKAkSize/LKAdilation)')
parser.add_argument('--DWDkSize', type=int, default=7,
                    help='kernel size of DepthWise dilation convolution. (default 7)')
parser.add_argument('--DWDdil', type=int, default=3,
                    help='dilation of DepthWise dilation convolution. (default 3)')
parser.add_argument('--no_attn', action='store_true',
                    help='do not use element-wise multiply as attention in LKA')
parser.add_argument('--bb_norm', default='BN',
                    choices=('BN', 'LN', 'no'),
                    help='norm type of backbone (BN | LN | no)')
parser.add_argument('--down_fea', action='store_true',
                    help='use feature down before extraction, and up after')

# DCAN settings
parser.add_argument('--use_wave', action='store_true',
                    help='use haar wavelet before extraction, and inverse after')
parser.add_argument('--wave_patchup', action='store_true',
                    help='expand patch size instead of dimension when use wavelet')

# ShuffleMixer settings
parser.add_argument('--ShMn_feats', type=int, default=64,
                    help='Number of channels. (Use in ShuffleMixer, base 64, tiny 32)')
parser.add_argument('--ShMkSize', type=int, default=7,
                    help='kernel size of Depthwise convolution. (Use in ShuffleMixer, base 7, tiny 3)')

# SwinIR Ablation model settings
parser.add_argument('--swinir_ablation', action='store_true',
                    help='use ablation settings to replace MSA in SwinIR with CNNs')
parser.add_argument('--swinir_mlpdwc', action='store_true',
                    help='use ablation settings to add DWConv in MLP of SwinIR')

args = parser.parse_args()
# add following means receive [] as input, replace the way of main.py [], to use in jupyter notebook
# args = parser.parse_args(args=[])
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.depths = list(map(lambda x: int(x), args.depths.split('+')))
args.dims = list(map(lambda x: int(x), args.dims.split('+')))
args.mlp_ratios = list(map(lambda x: float(x), args.mlp_ratios.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

# for multi-GPU training settings changing
if not args.cpu and args.n_GPUs > 1:
    # args.batch_size = args.batch_size * args.n_GPUs
    args.lr = args.lr * args.n_GPUs

if args.batch_size % args.accumulation_step != 0:
    raise ValueError('accumulation_step {} must divides into batch_size {}.'
                        .format(args.accumulation_step, args.batch_size))

if args.accumulation_step > 1:
    # choose one of following:
    args.batch_size = args.batch_size // args.accumulation_step  # maintain the same batch as setting
    # args.lr = args.lr * args.accumulation_step  # larger batch as multiply accumulation_step with larger lr

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

