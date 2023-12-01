import os
import math
from decimal import Decimal
import time

import utility
import model
from data import common

import torch
import torch.nn.utils as utils
import torch.nn.functional as F
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp, ref_model=None):
        self.args = args
        self.scale = args.scale
        self.accumulation_step = args.accumulation_step

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.ref_model = ref_model

        if self.args.load != '':
            self.optimizer.step()
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        accumulation_loss = 0
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            if (batch + 1) % self.accumulation_step == 0:
                self.optimizer.zero_grad()  # reset gradient
            if self.args.model == 'BISRCNN' or self.args.model == 'BICNNV2' \
                    or self.args.model == 'BIAANV3' or self.args.model == 'BIAANV3B' \
                    or self.args.model == 'BIAANV3D' or self.args.model == 'BIAANV9' \
                    or self.args.model == 'BIAANV9C' or self.args.model == 'BIAANV3H' \
                    or self.args.model == 'BIAANV10' or self.args.model == 'BIAANV12' \
                    or self.args.model == 'BIFSRCNNV3' or self.args.model == 'BIFSRCNNV6' \
                    or self.args.model == 'BIFSRCNNV9' or self.args.model == 'IRN':
                sr, br = self.model(lr, 0, hr)
                loss_forw = self.loss(sr, hr)
                loss_back = self.loss(br, lr)
                loss = (loss_forw + loss_back)/2
            elif self.args.model == 'BICNNV3' or self.args.model == 'BICNNV4' \
                    or self.args.model == 'BIFSRCNNV8' or self.args.model == 'BIFSRCNNV11':
                sr, br, ff1, ff2, bf1, bf2 = self.model(lr, 0, hr)
                loss_forw = self.loss(sr, hr)
                loss_back = self.loss(br, lr)
                loss_fea1 = self.loss(ff1, bf2)
                loss_fea2 = self.loss(ff2, bf1)
                loss = (loss_forw + loss_back + loss_fea1 + loss_fea2)/4
            elif self.args.model == 'BIAANV1' or self.args.model == 'BIAANV6' \
                    or self.args.model == 'BIAANV7' or self.args.model == 'BIAANV3C' \
                    or self.args.model == 'BIAANV8':# or self.args.model == 'BIAANV3D':
                sr, br, ff1, ff2, ff3, bf1, bf2, bf3 = self.model(lr, 0, hr)
                loss_fea1 = self.loss(ff1, bf3)
                loss_fea2 = self.loss(ff2, bf2)
                loss_fea3 = self.loss(ff3, bf1)
                loss_forw = self.loss(sr, hr)
                loss_back = self.loss(br, lr)
                loss = (loss_forw + loss_back + loss_fea1 + loss_fea2 + loss_fea3) / 5
            elif self.args.model == 'BIAAN' or self.args.model == 'BIAANV3E':  # BIAAN means version 5
                sr, br, ff1, ff2, bf1, bf2 = self.model(lr, 0, hr)
                loss_fea1 = self.loss(ff1, bf2)
                loss_fea2 = self.loss(ff2, bf1)
                loss_forw = self.loss(sr, hr)
                loss_back = self.loss(br, lr)
                loss = (loss_forw + loss_back + loss_fea1 + loss_fea2) / 4
            # elif self.args.model == 'BIAANV3F' or self.args.model == 'BIAANV9A' \
            #         or self.args.model == 'BIAANV3G' or self.args.model == 'BIAANV9B' \
            #         or self.args.model == 'BIAANV11':
            #     sr, br, ff1, ff2, bf2, bf3 = self.model(lr, 0, hr)
            #     ch = 40
            #     loss_fea1 = self.loss(ff1/ch, bf3/ch)
            #     loss_fea2 = self.loss(ff2/ch, bf2/ch)
            #     loss_forw = self.loss(sr, hr)
            #     loss_back = self.loss(br, lr)
            #     loss = (loss_forw + loss_back + loss_fea1 + loss_fea2) / 4
            elif self.args.model == 'BIAANV3F' or self.args.model == 'BIAANV9A' \
                    or self.args.model == 'BIAANV3G' or self.args.model == 'BIAANV9B' \
                    or self.args.model == 'BIAANV11':
                sr, br, ff1, ff2, bf2, bf3 = self.model(lr, 0, hr)
                loss_fea1 = self.loss(ff1, bf3)
                loss_fea2 = self.loss(ff2, bf2)
                loss_forw = self.loss(sr, hr)
                loss_back = self.loss(br, lr)
                loss = (loss_forw + loss_back + loss_fea1 + loss_fea2) / 4
            elif self.args.model == 'BIFSRCNN' or self.args.model == 'BIFSRCNNV4':
                sr, br, f1, f2, f3, f4, b1, b2, b3, b4 = self.model(lr, 0, hr)
                loss_forw = self.loss(sr, hr)
                loss_back = self.loss(br, lr)
                loss_fea1 = self.loss(f1, b4)
                loss_fea2 = self.loss(f2, b3)
                loss_fea3 = self.loss(f3, b2)
                loss_fea4 = self.loss(f4, b1)
                loss = (loss_forw + loss_back + loss_fea1 + loss_fea2 +
                        loss_fea3 + loss_fea4) / 6
            elif self.args.model == 'BIFSRCNNV2' or self.args.model == 'BIFSRCNNV5':
                sr, br, f1, f2, f3, f4, f5, f6, f7, b1, b2, b3, b4, b5, b6, b7 = self.model(lr, 0, hr)
                loss_forw = self.loss(sr, hr)
                loss_back = self.loss(br, lr)
                loss_fea1 = self.loss(f1, b7)
                loss_fea2 = self.loss(f2, b6)
                loss_fea3 = self.loss(f3, b5)
                loss_fea4 = self.loss(f4, b4)
                loss_fea5 = self.loss(f5, b3)
                loss_fea6 = self.loss(f6, b2)
                loss_fea7 = self.loss(f7, b1)
                loss = (loss_forw + loss_back + loss_fea1 + loss_fea2 + loss_fea3 +
                        loss_fea4 + loss_fea5 + loss_fea6 + loss_fea7) / 9
            elif self.args.model == 'BIFSRCNNV7' or self.args.model == 'BIFSRCNNV10' \
                    or self.args.model == 'BIFSRCNNPS' or self.args.model == 'BIFSRCNNPSV3' \
                    or self.args.model == 'BIFSRCNNLI' or self.args.model == 'BIFSRCNNLIV3' \
                    or self.args.model == 'BIFSRCNNLIV4':
                sr, br, f2, f3, f4, f5, f6, b2, b3, b4, b5, b6 = self.model(lr, 0, hr)
                loss_forw = self.loss(sr, hr)
                loss_back = self.loss(br, lr)
                loss_fea2 = self.loss(f2, b6)
                loss_fea3 = self.loss(f3, b5)
                loss_fea4 = self.loss(f4, b4)
                loss_fea5 = self.loss(f5, b3)
                loss_fea6 = self.loss(f6, b2)
                loss = (loss_forw + loss_back + loss_fea2 + loss_fea3 +
                        loss_fea4 + loss_fea5 + loss_fea6) / 7
                # loss = (loss_forw + loss_back + loss_fea2 + loss_fea6) / 4  # BIFSRCNNPSV3 & BIFSRCNNLIV2
            elif self.args.model == 'DFSRCNN' or self.args.model == 'DFSRCNNPS':
                br = self.model(hr, 0)
                loss = self.loss(br, lr)
            elif self.args.model == 'UFSRCNN':
                sr, f1, f2, f3, f4 = self.model(lr, 0, True)
                br, b1, b2, b3, b4 = self.ref_model(hr, 0, True)
                loss_forw = self.loss(sr, hr)
                # loss_fea1 = self.loss(f1, b4)  # v2
                loss_fea2 = self.loss(f2, b3)
                loss_fea3 = self.loss(f3, b2)
                # loss_fea4 = self.loss(f4, b1)  # v1
                # loss = (loss_forw + loss_fea1 + loss_fea2 + loss_fea3 + loss_fea4) / 5  # v1
                # loss = (loss_forw + loss_fea1 + loss_fea2 + loss_fea3) / 4  # v2
                loss = (loss_forw + loss_fea2 + loss_fea3) / 3
            elif self.args.model == 'UFSRCNNPS' or self.args.model == 'UFSRCNNPSV2' \
                    or self.args.model == 'UFSRCNNPSV6' or self.args.model == 'UFSRCNNPSV7':
                sr, f1, f2, f3, f4, f5, f6, f7 = self.model(lr, 0, True)
                br, b1, b2, b3, b4, b5, b6, b7 = self.ref_model(hr, 0, True)
                loss_forw = self.loss(sr, hr)
                loss_fea1 = self.loss(f1, b7)
                loss_fea2 = self.loss(f2, b6)
                # loss_fea3 = self.loss(f3, b5)
                # loss_fea4 = self.loss(f4, b4)
                # loss_fea5 = self.loss(f5, b3)
                loss_fea6 = self.loss(f6, b2)
                loss_fea7 = self.loss(f7, b1)
                # loss = (loss_forw + loss_fea2 + loss_fea3 + loss_fea4 + loss_fea5 + loss_fea6) / 6  # v2
                loss = (loss_forw + loss_fea2 + loss_fea6) / 3  # v3 & v7
                # loss = (loss_forw + loss_fea1 + loss_fea2 + loss_fea3 +
                #         loss_fea4 + loss_fea5 + loss_fea6 + loss_fea7) / 8  # v4
                # loss = (loss_forw + loss_fea1 + loss_fea2 + loss_fea6 + loss_fea7) / 5  # v5
            else:
                sr = self.model(lr, 0)
                loss = self.loss(sr, hr)
            loss = loss / self.accumulation_step  # use gradient accumulation
            accumulation_loss += loss.item()

            loss.backward()  # compute gradient

            
            # self.optimizer.step()  # update parameters
            if (batch + 1) % self.accumulation_step == 0:
                if accumulation_loss < self.args.skip_threshold * self.error_last:
                    if self.args.gclip > 0:
                        utils.clip_grad_value_(
                            self.model.parameters(),
                            self.args.gclip
                        )
                    self.optimizer.step()  # update parameters
                else:
                    print('Skip this batch {}! (Loss: {})'.format(
                        (batch + 1) // self.accumulation_step, loss.item()
                    ))
                # self.optimizer.zero_grad()  # reset gradient
            # if self.args.warmup:
            #     # if batch < len(self.loader_train)-1:
            #         with self.optimizer.warmup_scheduler.dampening():
            #             pass

            timer_model.hold()

            if (batch + 1) % self.accumulation_step == 0 and ((batch + 1) // self.accumulation_step) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    ((batch + 1) // self.accumulation_step) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                if self.args.model == 'DFSRCNN' or self.args.model == 'DFSRCNNPS':
                    for lr, hr, i_dim, filename in tqdm(d, ncols=80):
                        lr, hr = self.prepare(lr, hr, cb, cr)
                        br = self.model(hr, idx_scale)
                        br = utility.quantize(br, self.args.rgb_range)

                        save_list = [br]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            br, lr, scale, self.args.rgb_range, dataset=d
                        )
                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, scale)
                else:
                    if self.args.n_colors == 1:
                        for lr, hr, cb, cr, i_dim, filename in tqdm(d, ncols=80):
                            lr, hr, cb, cr = self.prepare(lr, hr, cb, cr)
                            sr = self.model(lr, idx_scale)
                            if self.args.runtime:
                                sum_run = 0
                                # sum_run_2 = 0
                                _times = self.args.times
                                for _i in range(_times):
                                    run_start = time.time()
                                    sr = self.model(lr, idx_scale)
                                    run_stop = time.time()
                                    runtime = run_stop - run_start
                                    self.ckp.write_log('No.{} Runtime: {:.2f}ms.'.format(_i + 1, runtime*1e3))
                                    sum_run += runtime
                                    # if _i != 0:
                                    #     sum_run_2 += runtime
                                self.ckp.write_log('Average Runtime: {:.2f}ms.\n'.format((sum_run/_times)*1e3))
                                # self.ckp.write_log('\nAverage Runtime except for first run: {:.2f}ms.'.format((sum_run_2/4)*1e3))
                            # else:
                            #     sr = self.model(lr, idx_scale)
                            sr = utility.quantize(sr, self.args.rgb_range)

                            save_list = [sr]
                            self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                                sr, hr, scale, self.args.rgb_range, dataset=d
                            )

                            ########################################################
                            # Add by CQP: if model output only Y channel, convert to RGB version.
                            if self.args.n_colors == 1 and i_dim == 3:
                                sr_cb = F.interpolate(cb, scale_factor=scale, mode='bicubic')
                                sr_cr = F.interpolate(cr, scale_factor=scale, mode='bicubic')
                                p_sr = [sr, sr_cb, sr_cr]
                                sr, sr_cb, sr_cr = [common.tensor2Np(i, rgb_range=self.args.rgb_range) for i in p_sr]
                                sr = common.y2rgb(sr[0], sr_cb[0], sr_cr[0])
                                p_sr = [sr]
                                p_sr_t = common.np2Tensor(*p_sr, rgb_range=self.args.rgb_range)
                                sr = p_sr_t[0].unsqueeze(0)
                                sr, = self.prepare(sr)
                                sr = utility.quantize(sr, self.args.rgb_range)
                                save_list.extend([sr])
                            ########################################################
                            if self.args.save_gt:
                                save_list.extend([lr, hr])

                            if self.args.save_results:
                                self.ckp.save_results(d, filename[0], save_list, scale)
                    else:
                        for lr, hr, filename in tqdm(d, ncols=80):
                            lr, hr = self.prepare(lr, hr)
                            # timer_run = utility.timer()
                            sr = self.model(lr, idx_scale)
                            if self.args.runtime:
                                sum_run = 0
                                # sum_run_2 = 0
                                _times = self.args.times
                                for _i in range(_times):
                                    run_start = time.time()
                                    sr = self.model(lr, idx_scale)
                                    run_stop = time.time()
                                    runtime = run_stop - run_start
                                    self.ckp.write_log('No.{} Runtime: {:.2f}ms.'.format(_i + 1, runtime*1e3))
                                    sum_run += runtime
                                    # if _i != 0:
                                    #     sum_run_2 += runtime
                                self.ckp.write_log('Average Runtime: {:.2f}ms.\n'.format((sum_run/_times)*1e3))
                                # self.ckp.write_log('\nAverage Runtime except for first run: {:.2f}ms.'.format((sum_run_2/4)*1e3))
                            # else:
                            #     sr = self.model(lr, idx_scale)
                            sr = utility.quantize(sr, self.args.rgb_range)

                            save_list = [sr]
                            self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                                sr, hr, scale, self.args.rgb_range, dataset=d
                            )

                            if self.args.save_gt:
                                save_list.extend([lr, hr])

                            if self.args.save_results:
                                self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                # best = self.ckp.log.max(0)
                best = torch.where(torch.isnan(self.ckp.log), torch.full_like(self.ckp.log, 0), self.ckp.log).max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
        if self.args.inf_switch:
            self.model.save_inf(self.ckp.get_path('model'), epoch)

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            # elif torch.backends.mps.is_available():  # torch 1.11 has no mps
            #     device = torch.device('mps')
            else:
                device = torch.device('cpu')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            if self.args.inf_switch:
                for m in self.model.modules():
                    if hasattr(m, 'switch_to_deploy'):
                        m.switch_to_deploy()
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
            # return epoch >= (self.args.epochs * self.accumulation_step)
