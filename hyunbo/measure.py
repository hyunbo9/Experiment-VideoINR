import os 
import math 
import time 
import glob

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def main():
    now = time.time()

    generated_folder_path = f'../output_scale_factor4/given_checkpoint/*/VideoINR'
    generated_folder_list = glob.glob(generated_folder_path) 

    original_folder_path = f'../adobe240/frame/test/*'
    original_folder_list = glob.glob(original_folder_path)

    result = {}
    for original_folder in original_folder_list:
        data_name = os.path.basename(original_folder)

        for checkpoint in ['given_checkpoint', 'default_checkpoint']:
            psnr_sum = 0
            ssim_sum = 0

            data_num = 0
            for original_img in glob.glob(f'{original_folder}/*.png'):
                original_img_name = os.path.basename(original_img)
                original_img_name = original_img_name.split('.')[0]
                original_img_name = int(original_img_name)

                generated_img = f'../output_scale_factor4/{checkpoint}/{data_name}/VideoINR/{original_img_name}.jpg'
                if not os.path.exists(generated_img):
                    #print("generated_img not exist")
                    #print(f"data_name: {data_name}, original_img_name: {original_img_name}")
                    continue
                data_num += 1

                original_img = cv2.imread(original_img)
                generated_img = cv2.imread(generated_img)

                # covert to yuv
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2YUV)
                generated_img = cv2.cvtColor(generated_img, cv2.COLOR_BGR2YUV)

                psnr_sum += psnr(original_img, generated_img)
                ssim_sum += ssim(original_img, generated_img)
                
            if checkpoint not in result:
                result[checkpoint] = {}
            else:
                result[checkpoint].update({
                    data_name: [psnr_sum / data_num, ssim_sum / data_num, data_num]
                    })

            #result[data_name + " from " + checkpoint] = f", psnr: {psnr_sum / data_num}, ssim: {ssim_sum / data_num}"
            print(" checkpoint: ", checkpoint, " data_name: ", data_name, " data_num: ", data_num, \
                        "psnr: ", psnr_sum / data_num, " ssim: ", ssim_sum / data_num)

        print("===================================================================================================")

    print("time: ", time.time() - now)

    """
    result = {
        checkpoint: {
            data_name: [psnr, ssim]
            }
        }
    """

    store_result_to_text(result, 'result_yuv_1223.txt')
    return result


# dict to store the results
def store_result_to_text(result, path):
    """
    result = {
        checkpoint: {
            data_name: [psnr, ssim]
            }
        }
    path: path to store the result
    """
    
    with open(path, 'w') as f:
        for checkpoint in ['given_checkpoint', 'default_checkpoint']:
            for data_name in result[checkpoint].keys():
                f.write(f"data_name: {data_name}, psnr: {result[checkpoint][data_name][0]}, ssim: {result[checkpoint][data_name][1]}, data_num: {result[checkpoint][data_name][2]} \n")
            f.write(f"=================================================================================================== \n")


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # print(img1)
    # print('img1-2')
    # print(img2)
    mse = np.mean((img1 - img2)**2)
    # print(mse)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



if __name__ == '__main__':
    main()