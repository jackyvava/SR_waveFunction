import random
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2


def gen_input_mask(shape, hole_size, hole_area=None, max_holes=1):
    """
    生成一个输入掩码，支持3D张量。
    * inputs:
        - shape (sequence, required):
                形状为 (N, C, D, H, W) 的张量，表示生成的掩码的形状。
        - hole_size (sequence or int, required):
                洞的大小。如果提供的是长度为2的序列，将生成大小为
                (D, H, W) = (
                    hole_size[0][0] <= hole_size[0][1],
                    hole_size[1][0] <= hole_size[1][1],
                    hole_size[2][0] <= hole_size[2][1],
                ) 的洞。
                洞内的所有像素值都填充为1.0。
        - hole_area (sequence, optional):
                限制生成洞的区域。hole_area[0] 是该区域的左上角 (X, Y, Z)，
                hole_area[1] 是其宽度、高度和深度 (W, H, D)。
                默认值为 None。
        - max_holes (int, optional):
                指定生成洞的数量。默认为1。
    * returns:
            形状为 [N, C, D, H, W] 的掩码张量。
            洞内的所有像素值都填充为1.0，而其他像素值为0.0。
    """
    mask = torch.zeros(shape)
    bsize, _, mask_d, mask_h, mask_w = mask.shape
    for i in range(bsize):
        n_holes = random.choice(list(range(1, max_holes + 1)))
        for _ in range(n_holes):
            # choose patch depth
            if isinstance(hole_size[0], tuple) and len(hole_size[0]) == 2:
                hole_d = random.randint(hole_size[0][0], hole_size[0][1])
            else:
                hole_d = hole_size[0]

            # choose patch width
            if isinstance(hole_size[1], tuple) and len(hole_size[1]) == 2:
                hole_w = random.randint(hole_size[1][0], hole_size[1][1])
            else:
                hole_w = hole_size[1]

            # choose patch height
            if isinstance(hole_size[2], tuple) and len(hole_size[2]) == 2:
                hole_h = random.randint(hole_size[2][0], hole_size[2][1])
            else:
                hole_h = hole_size[2]

            # choose offset upper-left coordinate
            if hole_area:
                harea_xmin, harea_ymin, harea_zmin = hole_area[0]
                harea_w, harea_h, harea_d = hole_area[1]
                offset_x = random.randint(harea_xmin, harea_xmin + harea_w - hole_w)
                offset_y = random.randint(harea_ymin, harea_ymin + harea_h - hole_h)
                offset_z = random.randint(harea_zmin, harea_zmin + harea_d - hole_d)
            else:
                offset_x = random.randint(0, mask_w - hole_w)
                offset_y = random.randint(0, mask_h - hole_h)
                offset_z = random.randint(0, mask_d - hole_d)
            mask[i, :, offset_z: offset_z + hole_d, offset_y: offset_y + hole_h, offset_x: offset_x + hole_w] = 1.0
    return mask


def gen_hole_area(size, mask_size):
    """
    生成洞的区域，支持3D张量。
    * inputs:
        - size (sequence, required):
                长度为3的序列 (W, H, D)，表示洞区域的大小。
        - mask_size (sequence, required):
                长度为3的序列 (W, H, D)，表示输入掩码的大小。
    * returns:
            用于生成掩码的 'hole_area' 参数。
    """
    mask_w, mask_h, mask_d = mask_size
    harea_w, harea_h, harea_d = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    offset_z = random.randint(0, mask_d - harea_d)
    return ((offset_x, offset_y, offset_z), (harea_w, harea_h, harea_d))


def crop(x, area):
    """
    裁剪3D张量中的特定区域。
    * inputs:
        - x (torch.Tensor, required)
                形状为 (N, C, D, H, W) 的3D张量。
        - area (sequence, required)
                长度为2的序列 ((X, Y, Z), (W, H, D))，指定要裁剪的区域。
    * returns:
            裁剪后的3D张量。
    """
    xmin, ymin, zmin = area[0]
    w, h, d = area[1]
    return x[:, :, zmin: zmin + d, ymin: ymin + h, xmin: xmin + w]


def sample_random_batch(dataset, batch_size=4):
    """
    从数据集中随机采样一个小批量数据，支持3D张量。
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                torch.utils.data.Dataset 的实例。
        - batch_size (int, optional)
                小批量的大小。默认为4。
    * returns:
            随机采样的小批量数据。
    """
    num_samples = len(dataset)
    batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)


def poisson_blend(input, output, mask):
    """
    使用泊松融合技术处理3D张量。
    * inputs:
        - input (torch.Tensor, required)
                Completion Network 的输入张量，形状为 (N, 3, D, H, W)。
        - output (torch.Tensor, required)
                Completion Network 的输出张量，形状为 (N, 3, D, H, W)。
        - mask (torch.Tensor, required)
                Completion Network 的输入掩码张量，形状为 (N, 1, D, H, W)。
    * returns:
                使用泊松图像编辑方法修复后的输出图像张量。
    """
    # 对于3D数据，这部分需要更复杂的处理方式，这里提供的是2D图像处理的代码。
    # 你可能需要修改为支持3D数据的处理流程，或者直接将处理改为基于图像的处理。
    input = input.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    mask = torch.cat((mask, mask, mask), dim=1)  # convert to 3-channel format
    num_samples = input.shape[0]
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(input[i, :, input.shape[2]//2])  # 使用中间切片
        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
        srcimg = transforms.functional.to_pil_image(output[i, :, output.shape[2]//2])
        srcimg = np.array(srcimg)[:, :, [2, 1, 0]]
        msk = transforms.functional.to_pil_image(mask[i, :, mask.shape[2]//2])
        msk = np.array(msk)[:, :, [2, 1, 0]]
        # 计算掩码中心
        xs, ys = [], []
        for j in range(msk.shape[0]):
            for k in range(msk.shape[1]):
                if msk[j, k, 0] == 255:
                    ys.append(j)
                    xs.append(k)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
        dstimg = cv2.inpaint(dstimg, msk[:, :, 0], 1, cv2.INPAINT_TELEA)
        out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
        out = out[:, :, [2, 1, 0]]
        out = transforms.functional.to_tensor(out)
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret
