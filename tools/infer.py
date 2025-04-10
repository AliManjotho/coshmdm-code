import copy
import os.path
import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning as L
import scipy.ndimage.filters as filters
from os.path import join as pjoin
from models import *
from collections import OrderedDict
from configs import get_config
from utils.plot_script import *
from utils.preprocess import *
from utils import paramUtil
import argparse

class LitGenModel(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # train model init
        self.model = model

        # others init
        self.normalizer = MotionNormalizer()

    def plot_t2m(self, mp_data, result_path, caption):
        mp_joint = []
        for i, data in enumerate(mp_data):
            if i == 0:
                joint = data[:,:22*3].reshape(-1,22,3)
            else:
                joint = data[:,:22*3].reshape(-1,22,3)

            mp_joint.append(joint)

        plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=30)


    def generate_one_sample(self, folder, prompt, name):
        self.model.eval()
        batch = OrderedDict({})

        batch["motion_lens"] = torch.zeros(1,1).long().cuda()
        batch["prompt"] = prompt

        window_size = 210
        motion_output = self.generate_loop(batch, window_size)
        result_path = f"results/{folder}/{name}.mp4"        

        if not os.path.exists(f"results/{folder}"):
            os.makedirs(f"results/{folder}")

        self.plot_t2m([motion_output[0], motion_output[1]],
                      result_path,
                      batch["prompt"])
        
        return motion_output[0], motion_output[1]

    def generate_loop(self, batch, window_size):
        prompt = batch["prompt"]
        batch = copy.deepcopy(batch)
        batch["motion_lens"][:] = window_size

        sequences = [[], []]

        batch["text"] = [prompt]
        batch = self.model.forward_test(batch)
        motion_output_both = batch["output"][0].reshape(batch["output"][0].shape[0], 2, -1)
        motion_output_both = self.normalizer.backward(motion_output_both.cpu().detach().numpy())


        for j in range(2):
            motion_output = motion_output_both[:,j]

            joints3d = motion_output[:,:22*3].reshape(-1,22,3)
            joints3d = filters.gaussian_filter1d(joints3d, 1, axis=0, mode='nearest')
            sequences[j].append(joints3d)


        sequences[0] = np.concatenate(sequences[0], axis=0)
        sequences[1] = np.concatenate(sequences[1], axis=0)
        return sequences

def build_models(cfg):
    if cfg.NAME == "CoShMDM":
        model = CoShMDM(cfg)
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument("--text_prompt", type=str, required=False, help='Prompt for generating interactions.')
    parser.add_argument("--text_file", type=str, required=False, default='.\prompts.txt', help='text file containing prompts for generating interactions.')
    parser.add_argument("--num_repetitions", type=int, required=False, default=3, help='repetition number to be rendered.')
    params = parser.parse_args()

    
    # torch.manual_seed(37)
    model_cfg = get_config("configs/model.yaml")
    infer_cfg = get_config("configs/infer.yaml")

    model = build_models(model_cfg)

    if model_cfg.CHECKPOINT:
        ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("checkpoint state loaded!")

    litmodel = LitGenModel(model, infer_cfg).to(torch.device("cuda:0"))



    if params.text_prompt != '':
        motions_p1 = []
        motions_p2 = []
        prompts = []
        lengths = []
        num_samples = 1
        num_repititions = params.num_repetitions


        name = params.text_prompt[:48].replace(' ', '_')
        for rep in range(num_repititions):
            m1, m2 = litmodel.generate_one_sample(name, params.text_prompt, str(rep))

            m1 = np.transpose(m1, (1, 2, 0))
            m2 = np.transpose(m2, (1, 2, 0))

            motions_p1.append(np.array(m1))
            motions_p2.append(np.array(m2))
            prompts.append(params.text_prompt)
            lengths.append(m1.shape[2])
                        

        
        motion_p1_data = {'motion': np.array(motions_p1),
                        'text': prompts,
                        'lengths': np.array(lengths),
                        'num_samples': 1,
                        'num_repititions': num_repititions}
        
        motion_p2_data = {'motion': np.array(motions_p2),
                        'text': prompts,
                        'lengths': np.array(lengths),
                        'num_samples': 1,
                        'num_repititions': num_repititions}
        
        np.save(f"results/{name}/p1.npy", motion_p1_data)
        np.save(f"results/{name}/p2.npy", motion_p2_data)



    elif params.text_file != '':
       

        with open(params.text_file) as f:
            texts = f.readlines()
        texts = [text.strip("\n") for text in texts]

        for text in texts:

            motions_p1 = []
            motions_p2 = []
            prompts = []
            lengths = []
            num_samples = 1
            num_repititions = params.num_repetitions


            name = text[:48].replace(' ', '_')
            for rep in range(num_repititions):
                m1, m2 = litmodel.generate_one_sample(name, text, str(rep))

                m1 = np.transpose(m1, (1, 2, 0))
                m2 = np.transpose(m2, (1, 2, 0))

                motions_p1.append(np.array(m1))
                motions_p2.append(np.array(m2))
                prompts.append(text)
                lengths.append(m1.shape[2])
                            

            
            motion_p1_data = {'motion': np.array(motions_p1),
                            'text': prompts,
                            'lengths': np.array(lengths),
                            'num_samples': 1,
                            'num_repititions': num_repititions}
            
            motion_p2_data = {'motion': np.array(motions_p2),
                            'text': prompts,
                            'lengths': np.array(lengths),
                            'num_samples': 1,
                            'num_repititions': num_repititions}
            
            np.save(f"results/{name}/p1.npy", motion_p1_data)
            np.save(f"results/{name}/p2.npy", motion_p2_data)


