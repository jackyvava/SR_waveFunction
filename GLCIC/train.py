import json
import os
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
import torch
import numpy as np
from tqdm import tqdm
from models import CompletionNetwork3D, ContextDiscriminator3D
from datasets import NpyDataset
from losses import completion_network_loss
from utils import (
    gen_input_mask,
    gen_hole_area,
    crop,
    sample_random_batch,
    poisson_blend,
)

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('result_dir')
parser.add_argument('--data_parallel', action='store_true')
parser.add_argument('--init_model_cn_a', type=str, default=None)
parser.add_argument('--init_model_cn_b', type=str, default=None)
parser.add_argument('--init_model_cd', type=str, default=None)
parser.add_argument('--steps_1', type=int, default=90000)
parser.add_argument('--steps_2', type=int, default=10000)
parser.add_argument('--steps_3', type=int, default=400000)
parser.add_argument('--snaperiod_1', type=int, default=10000)
parser.add_argument('--snaperiod_2', type=int, default=2000)
parser.add_argument('--snaperiod_3', type=int, default=10000)
parser.add_argument('--max_holes', type=int, default=1)
parser.add_argument('--hole_min_w', type=int, default=48)
parser.add_argument('--hole_max_w', type=int, default=96)
parser.add_argument('--hole_min_h', type=int, default=48)
parser.add_argument('--hole_max_h', type=int, default=96)
parser.add_argument('--cn_input_size', type=int, default=32)
parser.add_argument('--ld_input_size', type=int, default=16)
parser.add_argument('--bsize', type=int, default=4)
parser.add_argument('--bdivs', type=int, default=1)
parser.add_argument('--num_test_completions', type=int, default=4)
parser.add_argument('--mpv', nargs=3, type=float, default=None)
parser.add_argument('--alpha', type=float, default=4e-4)

def main(args):
    # ================================================
    # Preparation
    # ================================================
    if not torch.cuda.is_available():
        raise Exception('At least one gpu must be available.')
    gpu = torch.device('cuda:0')

    # create result directory (if necessary)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    for phase in ['phase_1_a', 'phase_1_b', 'phase_2', 'phase_3']:
        if not os.path.exists(os.path.join(args.result_dir, phase)):
            os.makedirs(os.path.join(args.result_dir, phase))

    # load dataset
    print('loading dataset... (it may take a few minutes)')
    train_dset_a = NpyDataset(
        velocity_dir=os.path.join(args.data_dir, 'train/velocities'),
        mask_ratio=0.2
    )
    train_dset_b = NpyDataset(
        velocity_dir=os.path.join(args.data_dir, 'train/velocities'),
        abcd_dir=os.path.join(args.data_dir, 'train/abcd'),
        mask_ratio=0.2
    )
    test_dset_a = NpyDataset(
        velocity_dir=os.path.join(args.data_dir, 'test/velocities'),
        mask_ratio=0.2
    )
    test_dset_b = NpyDataset(
        velocity_dir=os.path.join(args.data_dir, 'test/velocities'),
        abcd_dir=os.path.join(args.data_dir, 'test/abcd'),
        mask_ratio=0.2
    )

    train_loader_a = DataLoader(
        train_dset_a,
        batch_size=(args.bsize // args.bdivs),
        shuffle=True
    )
    train_loader_b = DataLoader(
        train_dset_b,
        batch_size=(args.bsize // args.bdivs),
        shuffle=True
    )

    # compute mpv (mean pixel value) of training dataset
    if args.mpv is None:
        mpv = np.zeros(shape=(3,))
        pbar = tqdm(total=len(train_dset_a))
        for velocity in train_dset_a:
            mpv += velocity.mean(axis=(1, 2, 3))  # 修改为3D的均值计算
            pbar.update()
        mpv /= len(train_dset_a)
        pbar.close()
    else:
        mpv = np.array(args.mpv)

    mpv = torch.tensor(mpv.reshape(1, 3, 1, 1, 1), dtype=torch.float32).to(gpu)  # 更新为3D形状
    alpha = torch.tensor(args.alpha, dtype=torch.float32).to(gpu)

    # ================================================
    # Training Phase 1: Model A
    # ================================================
    model_cn_a = CompletionNetwork3D()
    if args.init_model_cn_a is not None:
        model_cn_a.load_state_dict(torch.load(
            args.init_model_cn_a,
            map_location='cpu'))
    if args.data_parallel:
        model_cn_a = DataParallel(model_cn_a)
    model_cn_a = model_cn_a.to(gpu)
    opt_cn_a = Adadelta(model_cn_a.parameters())

    # Training loop for Model A
    pbar = tqdm(total=args.steps_1)
    while pbar.n < args.steps_1:
        for x in train_loader_a:
            x = x.to(gpu)
            mask = gen_input_mask(
                shape=(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]),
                hole_size=(
                    (args.hole_min_w, args.hole_max_w),
                    (args.hole_min_h, args.hole_max_h)),
                hole_area=gen_hole_area(
                    (args.ld_input_size, args.ld_input_size),
                    (x.shape[3], x.shape[2])),
                max_holes=args.max_holes
            ).to(gpu)
            x_mask = x - x * mask + mpv * mask
            input = torch.cat((x_mask, mask), dim=1)
            output = model_cn_a(input)
            loss = completion_network_loss(x, output, mask)

            # backward and optimize
            loss.backward()
            opt_cn_a.step()
            opt_cn_a.zero_grad()
            pbar.set_description('Model A | phase 1 | train loss: %.5f' % loss.cpu())
            pbar.update()

            # test and save model
            if pbar.n % args.snaperiod_1 == 0:
                test_and_save_model(model_cn_a, test_dset_a, 'phase_1_a', pbar.n, args.result_dir, mpv, args)
            if pbar.n >= args.steps_1:
                break
    pbar.close()

    # ================================================
    # Training Phase 1: Model B
    # ================================================
    model_cn_b = CompletionNetwork3D()
    if args.init_model_cn_b is not None:
        model_cn_b.load_state_dict(torch.load(
            args.init_model_cn_b,
            map_location='cpu'))
    if args.data_parallel:
        model_cn_b = DataParallel(model_cn_b)
    model_cn_b = model_cn_b.to(gpu)
    opt_cn_b = Adadelta(model_cn_b.parameters())

    # Training loop for Model B
    pbar = tqdm(total=args.steps_1)
    while pbar.n < args.steps_1:
        for masked_velocity, abcd, targets in train_loader_b:
            masked_velocity, abcd, targets = masked_velocity.to(gpu), abcd.to(gpu), targets.to(gpu)
            mask = gen_input_mask(
                shape=(masked_velocity.shape[0], 1, masked_velocity.shape[2], masked_velocity.shape[3], masked_velocity.shape[4]),
                hole_size=(
                    (args.hole_min_w, args.hole_max_w),
                    (args.hole_min_h, args.hole_max_h)),
                hole_area=gen_hole_area(
                    (args.ld_input_size, args.ld_input_size),
                    (masked_velocity.shape[3], masked_velocity.shape[2])),
                max_holes=args.max_holes
            ).to(gpu)
            x_mask = masked_velocity - masked_velocity * mask + mpv * mask
            input = torch.cat((x_mask, mask, abcd), dim=1)
            output = model_cn_b(input)
            loss = completion_network_loss(targets, output, mask)

            # backward and optimize
            loss.backward()
            opt_cn_b.step()
            opt_cn_b.zero_grad()
            pbar.set_description('Model B | phase 1 | train loss: %.5f' % loss.cpu())
            pbar.update()

            # test and save model
            if pbar.n % args.snaperiod_1 == 0:
                test_and_save_model(model_cn_b, test_dset_b, 'phase_1_b', pbar.n, args.result_dir, mpv, args)
            if pbar.n >= args.steps_1:
                break
    pbar.close()

    # ================================================
    # Training Phase 2 and 3 for Context Discriminator (如有需要)
    # ================================================
    # Similar process as above but for ContextDiscriminator
    # Use the model_cd, model_cn_a, and model_cn_b
    # ...

def test_and_save_model(model, test_dset, phase, step, result_dir, mpv, args):
    model.eval()
    with torch.no_grad():
        x = sample_random_batch(test_dset, batch_size=args.num_test_completions).to(gpu)
        mask = gen_input_mask(
            shape=(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]),
            hole_size=(
                (args.hole_min_w, args.hole_max_w),
                (args.hole_min_h, args.hole_max_h)),
            hole_area=gen_hole_area(
                (args.ld_input_size, args.ld_input_size),
                (x.shape[3], x.shape[2])),
            max_holes=args.max_holes
        ).to(gpu)
        x_mask = x - x * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        output = model(input)
        completed = poisson_blend(x_mask, output, mask)
        npy_path = os.path.join(result_dir, phase, 'step%d.npy' % step)
        np.save(npy_path, completed.cpu().numpy())
        model_path = os.path.join(result_dir, phase, 'model_step%d' % step)
        torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    args.result_dir = os.path.expanduser(args.result_dir)
    if args.init_model_cn_a is not None:
        args.init_model_cn_a = os.path.expanduser(args.init_model_cn_a)
    if args.init_model_cn_b is not None:
        args.init_model_cn_b = os.path.expanduser(args.init_model_cn_b)
    if args.init_model_cd is not None:
        args.init_model_cd = os.path.expanduser(args.init_model_cd)
    main(args)
