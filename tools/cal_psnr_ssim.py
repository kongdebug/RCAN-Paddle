import cv2
import numpy as np
import os
import re
import math
from PIL import Image
import argparse

def reorder_image(img, input_order='HWC'):
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def rgb2ycbcr(img, y_only=False):
    img_type = img.dtype

    if img_type != np.uint8:
        img *= 255.

    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) / 255. + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]

    if img_type != np.uint8:
        out_img /= 255.
    else:
        out_img = out_img.round()

    return out_img

def bgr2ycbcr(img, y_only=False):
    img_type = img.dtype

    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]
    return out_img

def to_y_channel(img):
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = rgb2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.

def calculate_psnr(img1,
                   img2,
                   crop_border = 0,
                   input_order='HWC',
                   test_y_channel=True):

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = img1.copy().astype('float32')
    img2 = img2.copy().astype('float32')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))

def _ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1,
                   img2,
                   crop_border=0,
                   input_order='HWC',
                   test_y_channel=True):

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    img1 = img1.copy().astype('float32')[..., ::-1]
    img2 = img2.copy().astype('float32')[..., ::-1]

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt_dir", type=str, 
                        default="data/Set14/GTmod12")
    parser.add_argument("--output_dir", type=str,
                        default="output_dir/rcan_x4_div2k-2021-11-27-17-18/visual_test")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # gt_dir = r"data/Set5/GTmod12"
    # sr_dir = r"output_dir/rcan_x4_div2k-2021-11-25-13-30/visual_test"
    gt_dir = args.gt_dir
    sr_dir = args.output_dir
    filepaths = os.listdir(gt_dir)
    num_files = len(filepaths)
    psnr = []
    ssmi = []
    num = 0
    # file_handle = open(r'C:\Users\58381\Desktop\result.txt',mode='a')
    # file_handle.writelines(sr_dir+'\n')
    for i in range(num_files):
        gt_name = filepaths[i]
        img_name, extension = os.path.splitext(gt_name)
        print(num)
        num = num+1
        if extension not in ['.tiff','.png']:
            continue

        gt_img = cv2.imread(os.path.join(gt_dir,gt_name))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        #gt_img = Image.open(os.path.join(gt_dir,gt_name)).convert('RGB')
        
        img_name = re.sub('_gt', '', img_name)

        #sr_name = "iter"+"238000_"+img_name+"_output"+extension

        sr_name = img_name+'_output.png'
        sr_img = cv2.imread(os.path.join(sr_dir, sr_name))

        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)

        #sr_img = Image.open(os.path.join(sr_dir,sr_name)).convert('RGB')

        ps = calculate_psnr(sr_img, gt_img,4)
        ss = calculate_ssim(sr_img, gt_img,4)
        # file_handle.writelines([sr_name+"\t",str(ps)+'\t',str(ss)+'\n'])
        #ss = calculate_ssim(gt_img, sr_img)
        #ps = PSNR(gt_img, sr_img)
        psnr.append(ps)
        ssmi.append(ss)
    
    avg_psnr = np.sum(np.array(psnr))/len(psnr)
    avg_ssim = np.sum(np.array(ssmi))/len(ssmi)
    # file_handle.close()
    print(avg_psnr,avg_ssim)




