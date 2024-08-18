import os
import argparse
import json
import torch
import numpy as np
from models import CompletionNetwork3D  # 使用3D的CompletionNetwork
from utils import poisson_blend, gen_input_mask

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config')
parser.add_argument('input_npy')
parser.add_argument('output_npy')
parser.add_argument('--max_holes', type=int, default=5)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=24)
parser.add_argument('--hole_max_w', type=int, default=48)
parser.add_argument('--hole_min_h', type=int, default=24)
parser.add_argument('--hole_max_h', type=int, default=48)

def main(args):

    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.input_npy = os.path.expanduser(args.input_npy)
    args.output_npy = os.path.expanduser(args.output_npy)

    # =============================================
    # Load model
    # =============================================
    with open(args.config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1, 1)  # 3D数据形状
    model = CompletionNetwork3D()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    # =============================================
    # Predict
    # =============================================
    # 读取并转换npy数据
    x = np.load(args.input_npy)
    x = torch.from_numpy(x).unsqueeze(0)  # 增加batch维度

    # 创建掩码
    mask = gen_input_mask(
        shape=(1, 1, x.shape[2], x.shape[3], x.shape[4]),  # 3D形状
        hole_size=(
            (args.hole_min_w, args.hole_max_w),
            (args.hole_min_h, args.hole_max_h),
        ),
        max_holes=args.max_holes,
    )

    # 修复
    model.eval()
    with torch.no_grad():
        x_mask = x - x * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        output = model(input)
        inpainted = poisson_blend(x_mask, output, mask)
        
        # 保存修复后的npy文件
        inpainted_np = inpainted.squeeze(0).cpu().numpy()  # 移除batch维度
        np.save(args.output_npy, inpainted_np)
    print('Output npy was saved as %s.' % args.output_npy)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
