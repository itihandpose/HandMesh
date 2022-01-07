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
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
from cmr.cmr_sg import CMR_SG
from cmr.cmr_pg import CMR_PG
from cmr.cmr_g import CMR_G
from mobrecon.mobrecon_densestack import MobRecon
from utils.read import spiral_tramsform
from utils import utils, writer
from options.base_options import BaseOptions
from datasets.FreiHAND.freihand import FreiHAND
from datasets.Human36M.human36m import Human36M
from torch.utils.data import DataLoader
from run2 import Runner
from termcolor import cprint
from tensorboardX import SummaryWriter
import sys
from streamlit import cli as stcli


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

def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    buffer = st.file_uploader("Image here")
    temp_file = NamedTemporaryFile(delete=False)
    my_image = cv2.imread("images/64_img.jpg")
    image = cv2.resize(my_image, (args.size, args.size))
    frame = runner.demo(image)
    # st.write(my_image)
    cv2.imshow('image', frame)
    cv2.waitKey(0)
        
if __name__ == '__main__':
        if st._is_running_with_streamlit:
            main()
        else:
            sys.argv = ["streamlit", "run", sys.argv[0]]
            sys.exit(stcli.main())
