import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import models.modules.Sakuya_arch as Sakuya_arch

from pdb import set_trace as bp
from data.util import imresize_np

parser = argparse.ArgumentParser()
parser.add_argument('--space_scale', type=int, default=4, help="upsampling space scale")
parser.add_argument('--time_scale', type=int, default=8, help="upsampling time scale")
#parser.add_argument('--data_path', type=str, required=True, help="data path for testing")
parser.add_argument('--data_folder', type=str, required=True, help="data folder for testing")

#parser.add_argument('--out_path_lr', type=str, default="./output/LR/", help="output path (Low res image)")
#parser.add_argument('--out_path_bicubic', type=str, default="./output/Bicubic/", help="output path (bicubic upsampling)")
#parser.add_argument('--out_path_ours', type=str, default="./output/VideoINR/", help="output path (VideoINR)")
parser.add_argument('--model_path', type=str, default="latest_G.pth", help="model parameter path")

opt = parser.parse_known_args()[0]


def single_forward(model, imgs_in, space_scale, time_scale, device):
    with torch.no_grad():
        b, n, c, h, w = imgs_in.size()
        h_n = int(4 * np.ceil(h / 4))
        w_n = int(4 * np.ceil(w / 4))
        imgs_temp = imgs_in.new_zeros(b, n, c, h_n, w_n)
        imgs_temp[:, :, :, 0:h, 0:w] = imgs_in

        time_Tensors = [torch.tensor([i / time_scale], device=device)[None].to(device) for i in range(time_scale)]
        #import pdb; pdb.set_trace()
        model_output = model(imgs_temp, time_Tensors, space_scale, test=True)

        return model_output


folder_list = os.listdir(opt.data_folder)
given_checkpoint_path = "latest_G.pth"
defalut_checkpoint_path = "saved_checkpoints/latest_G.pth"

device = 'cuda:0'
model = Sakuya_arch.LunaTokis(64, 6, 8, 5, 40)
model.load_state_dict(torch.load(opt.model_path), strict=True)

output_folder = 'output_scale_factor4'

for check_point_path in [given_checkpoint_path, defalut_checkpoint_path]:
    model.load_state_dict(torch.load(check_point_path), strict=True)
    model.eval()
    model = model.to(device)

    for folder in folder_list:
        folder_path = os.path.join(opt.data_folder, folder)
        if check_point_path == given_checkpoint_path:
            save_path = os.path.join(f"./{output_folder}/given_checkpoint", folder)
            break
        elif check_point_path == defalut_checkpoint_path:
            save_path = os.path.join(f"./{output_folder}/default_checkpoint", folder)
            
        else:
            raise Error("check_point_path is not valid")
    
        os.makedirs(save_path, exist_ok=True)

        LR_folder = os.path.join(save_path, "LR")
        Bicubic_folder = os.path.join(save_path, "Bicubic")
        VideoINR_folder = os.path.join(save_path, "VideoINR")

        os.makedirs(LR_folder, exist_ok=True)
        os.makedirs(Bicubic_folder, exist_ok=True)
        os.makedirs(VideoINR_folder, exist_ok=True)

        path_list = [os.path.join(folder_path, name) for name in sorted(os.listdir(folder_path))]
        index = 0

        # print(*path_list, sep="\n")
        # sys.exit()

        for ind in tqdm(range(len(path_list) - 1)):

            imgpath1 = os.path.join(path_list[ind])
            imgpath2 = os.path.join(path_list[ind + 1])

            if '.DS_Store' in imgpath1:
                continue 

            img1 = cv2.imread(imgpath1, cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread(imgpath2, cv2.IMREAD_UNCHANGED)

            '''
            We apply down-sampling on the original video
            in order to avoid CUDA out of memory.
            You may skip this step if your input video
            is already of relatively low resolution.
            '''
            #import pdb;pdb.set_trace()
            scale_factor = 4
            
            img1 = imresize_np(img1, 1 / scale_factor, True).astype(np.float32) / 255.
            img2 = imresize_np(img2, 1 / scale_factor, True).astype(np.float32) / 255.

            #import pdb;pdb.set_trace()

            Image.fromarray((np.clip(img1[:, :, [2, 1, 0]], 0, 1) * 255).astype(np.uint8)).save(
                os.path.join(LR_folder, path_list[ind].split('/')[-1]))

            imgs = np.stack([img1, img2], axis=0)[:, :, :, [2, 1, 0]]
            imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()[None].to(device)

            output = single_forward(model, imgs, opt.space_scale, opt.time_scale, device)

            '''
            Save results of VideoINR and bicubic up-sampling.
            '''

            for out_ind in range(len(output)):

                img = output[out_ind][0]
                img = Image.fromarray((img.clamp(0., 1.).detach().cpu().permute(1, 2, 0) * 255).numpy().astype(np.uint8))
                img.save(os.path.join(VideoINR_folder, '{}.jpg'.format(index)))

                HH, WW = img1.shape[0] * 4, img1.shape[1] * 4
                img = Image.fromarray((np.clip(img1[:, :, [2, 1, 0]], 0, 1) * 255).astype(np.uint8)).resize((WW, HH),Image.BICUBIC)
                img.save(os.path.join(Bicubic_folder, '{}.jpg'.format(index)))
                index += 1
