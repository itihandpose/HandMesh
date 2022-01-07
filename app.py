import json
import os
import os.path as osp
import pickle
import time
from tempfile import NamedTemporaryFile

import cv2
import imageio
import numpy as np
import streamlit as st
import torch
from PIL import Image
from termcolor import colored, cprint

from datasets.FreiHAND.kinematics import mano_to_mpii
from mobrecon.mobrecon_densestack import MobRecon
from options.base_options_2 import BaseOptions
from utils import utils, writer
from utils.draw3d import (draw_2d_skeleton, draw_3d_skeleton,
                          save_a_image_with_mesh_joints)
from utils.progress.bar import Bar
from utils.read import save_mesh, spiral_tramsform
from utils.transforms import rigid_align
from utils.vis import (base_transform, cnt_area, inv_base_tranmsform, map2uv,
                       registration, tensor2array)


class Runner(object):
    def __init__(self, args, model, faces, device):
        super(Runner, self).__init__()
        self.args = args
        self.model = model
        self.faces = faces
        self.device = device
        self.face = torch.from_numpy(self.faces[0].astype(np.int64)).to(self.device)

    def set_demo(self, args):
      with open(os.path.join(args.work_dir, 'template', 'MANO_RIGHT.pkl'), 'rb') as f:
          mano = pickle.load(f, encoding='latin1')
      self.j_regressor = np.zeros([21, 778])
      self.j_regressor[:16] = mano['J_regressor'].toarray()
      for k, v in {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}.items():
          self.j_regressor[k, v] = 1
      self.std = torch.tensor(0.20)

    def poseEstimator(self, image):
        args = self.args
        self.model.eval()

        with torch.no_grad():
            # image = Image.fromarray(np.uint8(image)) #.convert('RGB')
            # image.resize(size=(args.size, args.size))
            # image = np.array(image)
            input = torch.from_numpy(base_transform(image, size=args.size)).unsqueeze(0).to(self.device) # A tensor with shape (1, 128, 128, 3)
            K = np.array([[500, 0, 128], [0, 500, 128], [0, 0, 1]])
                
            K[0, 0] = K[0, 0] / 224 * args.size
            K[1, 1] = K[1, 1] / 224 * args.size
            K[0, 2] = args.size // 2
            K[1, 2] = args.size // 2

            out = self.model(input)

            # silhouette
            mask_pred = out.get('mask_pred')
            if mask_pred is not None:
                mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
                mask_pred = cv2.resize(mask_pred, (input.size(3), input.size(2)))
                try:
                    contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours.sort(key=cnt_area, reverse=True)
                    poly = contours[0].transpose(1, 0, 2).astype(np.int32)
                except:
                    poly = None
            else:
                mask_pred = np.zeros([input.size(3), input.size(2)])
                poly = None

                
            # vertex
            pred = out['mesh_pred'][0] if isinstance(out['mesh_pred'], list) else out['mesh_pred']

            vertex = (pred[0].cpu() * self.std.cpu()).numpy() # Shape (778, 3)
                
            uv_pred = out['uv_pred']
            if uv_pred.ndim == 4:
                uv_point_pred, uv_pred_conf = map2uv(uv_pred.cpu().numpy(), (input.size(2), input.size(3)))
            else:
                uv_point_pred, uv_pred_conf = (uv_pred * args.size).cpu().numpy(), [None,]
            vertex, align_state = registration(vertex, uv_point_pred[0], self.j_regressor, K, args.size, uv_conf=uv_pred_conf[0], poly=poly)

            vertex2xyz = mano_to_mpii(np.matmul(self.j_regressor, vertex))
            print("Vertices: ", uv_point_pred[0])
            skeleton_overlay = draw_2d_skeleton(image[..., ::-1], uv_point_pred[0])
            frame = skeleton_overlay[..., ::-1]
        return frame

# get config
args = BaseOptions().parse()

# dir prepare
args.work_dir = osp.dirname(osp.realpath(__file__))
data_fp = osp.join(args.work_dir, 'data', args.dataset)
args.out_dir = osp.join(args.work_dir, 'out', args.dataset, args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
utils.makedirs(osp.join(args.out_dir, args.phase))
utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

template_fp = osp.join(args.work_dir, 'template', 'template.ply')
transform_fp = osp.join(args.work_dir, 'template', 'transform.pkl')
spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, args.ds_factors, args.seq_length, args.dilation)

for i in range(len(up_transform_list)):
    up_transform_list[i] = (*up_transform_list[i]._indices(), up_transform_list[i]._values())
    model = MobRecon(args, spiral_indices_list, up_transform_list)

device = torch.device('cpu')
torch.set_num_threads(args.n_threads)
runner = Runner(args, model, tmp['face'], device)
runner.set_demo(args)

st.set_option('deprecation.showfileUploaderEncoding', False)

buffer = st.file_uploader("Upload or drop image here")
temp_file = NamedTemporaryFile(delete=False)
# if buffer:
# temp_file.write(buffer.getvalue())
    # my_image= imageio.imread(temp_file.name)
    # print("type: ", type(my_image))
    # print("value: ", my_image)
my_image = cv2.imread("images/64_img.jpg")
image = cv2.resize(my_image, (args.size, args.size))
frame = runner.poseEstimator(image)
# st.write(my_image)
cv2.imshow('image', frame)
cv2.waitKey(0)








# image_fp = os.path.join(args.work_dir, 'images')
# image_files = [os.path.join(image_fp, i) for i in os.listdir(image_fp) if '_img.jpg' in i]
# for step, image_path in enumerate(image_files):
#     image_name = image_path.split('/')[-1].split('_')[0]
#     image = cv2.imread(image_path)#[..., ::-1] # Reverse RGB to BGR
#     image = cv2.resize(image, (args.size, args.size))
#     frame = runner.poseEstimator(image)
#     cv2.imshow('my webcam', frame)
#     if cv2.waitKey(1) == 27: 
#         break  # esc to quit
#     cv2.waitKey(0)
# cv2.destroyAllWindows()
