import argparse
import os
import visualize.vis_utils as vis_utils
import shutil
from tqdm import tqdm
import torch
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='path to folder containing npy files.')
    parser.add_argument("--repetition_num", type=int, required=True, default=0, help='repetition number to be rendered.')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()


    out_npy_path_p1 = os.path.join(params.input_path, f'p1_{params.repetition_num}_smpl_params.npy')
    out_npy_path_p2 = os.path.join(params.input_path, f'p2_{params.repetition_num}_smpl_params.npy')
    

    npy_path_p1 = os.path.join(params.input_path, 'p1.npy')
    npy_path_p2 = os.path.join(params.input_path, 'p2.npy')

    results_dir = os.path.join(params.input_path, 'objs_' + str(params.repetition_num))

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    objs_path_p1 = os.path.join(results_dir, 'p1')
    objs_path_p2 = os.path.join(results_dir, 'p2')

    if os.path.exists(objs_path_p1):
        shutil.rmtree(objs_path_p1)
    os.makedirs(objs_path_p1)

    if os.path.exists(objs_path_p2):
        shutil.rmtree(objs_path_p2)
    os.makedirs(objs_path_p2)

    sample_num = 0
    npy2obj_p1 = vis_utils.npy2obj(npy_path_p1, sample_num, params.repetition_num,
                                device=params.device, cuda=params.cuda)
    
    npy2obj_p2 = vis_utils.npy2obj(npy_path_p2, sample_num, params.repetition_num,
                                device=params.device, cuda=params.cuda)
    
    print('Saving obj files for Person 1 [{}]'.format(os.path.abspath(objs_path_p1)))
    for frame_i in tqdm(range(npy2obj_p1.real_num_frames)):
        npy2obj_p1.save_obj(os.path.join(objs_path_p1, 'p1_frame{:03d}.obj'.format(frame_i)), frame_i)

    print('Saving obj files for Person 2 [{}]'.format(os.path.abspath(objs_path_p2)))
    for frame_i in tqdm(range(npy2obj_p2.real_num_frames)):
        npy2obj_p2.save_obj(os.path.join(objs_path_p2, 'p2_frame{:03d}.obj'.format(frame_i)), frame_i)

    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path_p1)))
    npy2obj_p1.save_npy(out_npy_path_p1)

    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path_p2)))
    npy2obj_p2.save_npy(out_npy_path_p2)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_path", type=str, required=True, help='stick figure mp4 file to be rendered.')
#     parser.add_argument("--cuda", type=bool, default=True, help='')
#     parser.add_argument("--device", type=int, default=0, help='')
#     params = parser.parse_args()

#     assert params.input_path.endswith('.mp4')
#     parsed_name = os.path.basename(params.input_path).replace('.mp4', '').replace('prompt', '').replace('rep', '')
#     sample_i, rep_i = [int(e) for e in parsed_name.split('_')]
#     npy_path = os.path.join(os.path.dirname(params.input_path), 'motions.npy')
#     out_npy_path = params.input_path.replace('.mp4', '_smpl_params.npy')
#     assert os.path.exists(npy_path)
#     results_dir = params.input_path.replace('.mp4', '_obj')
#     if os.path.exists(results_dir):
#         shutil.rmtree(results_dir)
#     os.makedirs(results_dir)

#     npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
#                                 device=params.device, cuda=params.cuda)

#     print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
#     for frame_i in tqdm(range(npy2obj.real_num_frames)):
#         npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)

#     print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
#     npy2obj.save_npy(out_npy_path)
