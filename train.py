#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import copy
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render,render2, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import argparse
import options
import utils
from dataset.dataset_motiondeblur import *
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR,LambdaLR
from timm.utils import NativeScaler
from losses import CharbonnierLoss
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx
import math
from losses import CharbonnierLoss
from torchvision.transforms import ToPILImage
import Spline
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
def expand2square(timg,factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    
    return img, mask
class CustomDictionary:
    def __init__(self):
        self.data = {}

    def add_entry(self, key_tensor, value_vector):

        key_tuple = tuple(map(tuple, key_tensor.tolist()))
        self.data[key_tuple] = value_vector

    def get_values(self, key_tensor):
        key_tuple = tuple(map(tuple, key_tensor.tolist()))
        return self.data.get(key_tuple, None)
    
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_opt_iter=0
    differ=0
    torch.autograd.set_detect_anomaly(True)
    gaussian_dict = CustomDictionary()



    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    viewpoint_stack0=scene.getTrainCameras().copy()
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    idx=0

    for iteration in range(first_iter, opt.iterations + 1): 
        
        # original_images=[]      
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()


        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        if not viewpoint_stack:
            viewpoint_stack = viewpoint_stack0.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

    

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.restored_image.cuda()
        Ll1_A = l1_loss(image, gt_image)
        loss1 = (1.0 - opt.lambda_dssim) * Ll1_A + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss1.backward(retain_graph=True)
        gaussians_copy = copy.deepcopy(gaussians)


        iter_end.record()
        if (iteration in saving_iterations):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)
        if(differ<0):
            print("iteration",iteration,"differ",differ)
            stack = []
            gaussian_clone = []
            gaussian_split = []
            clone_vector=[]
            split_vector=[]
            clone_opt_vector=[]
            split_opt_vector=[]
            vnum=0

            for v in viewpoint_stack0:
                pipe.debug = False
                bg = torch.rand((3), device="cuda") if opt.random_background else background
                render_pkg = render(v, gaussians_copy, pipe, bg)
                image_copy, viewspace_point_tensor_copy, visibility_filter_copy, radii_copy = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                gt_image_copy = v.restored_image.cuda()
                Ll_A_copy = l1_loss(image, gt_image)
                loss_copy = (1.0 - opt.lambda_dssim) * Ll_A_copy + opt.lambda_dssim * (1.0 - ssim(image_copy, gt_image_copy))
                loss_copy.backward(retain_graph=True)
                


                gaussians_copy.max_radii2D[visibility_filter_copy] = torch.max(gaussians_copy.max_radii2D[visibility_filter_copy], radii[visibility_filter_copy])
                gaussians_copy.add_densification_stats(viewspace_point_tensor_copy, visibility_filter_copy)

                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                clone_mask_copy, split_mask_copy = gaussians_copy.densify_and_prune_test(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                stack.append((v.world_view_transform, clone_mask_copy, split_mask_copy))            
        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            if(first_opt_iter==0):
                first_opt_iter=iteration
            stack = []
            gaussian_clone = []
            gaussian_split = []
            clone_vector=[]
            split_vector=[]
            clone_opt_vector=[]
            split_opt_vector=[]
            clone_des_vector=[]
            split_des_vector=[]
            vnum=0

            for v in viewpoint_stack0:
                pipe.debug = False
                bg = torch.rand((3), device="cuda") if opt.random_background else background
                render_pkg = render(v, gaussians_copy, pipe, bg)
                image_copy, viewspace_point_tensor_copy, visibility_filter_copy, radii_copy = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                gt_image_copy = v.restored_image.cuda()
                Ll_A_copy = l1_loss(image, gt_image)
                loss_copy = (1.0 - opt.lambda_dssim) * Ll_A_copy + opt.lambda_dssim * (1.0 - ssim(image_copy, gt_image_copy))
                loss_copy.backward(retain_graph=True)
                


                gaussians_copy.max_radii2D[visibility_filter_copy] = torch.max(gaussians_copy.max_radii2D[visibility_filter_copy], radii[visibility_filter_copy])
                gaussians_copy.add_densification_stats2(viewspace_point_tensor_copy, visibility_filter_copy)

                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                clone_mask_copy, split_mask_copy = gaussians_copy.densify_and_prune_test(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                stack.append((v.world_view_transform, clone_mask_copy, split_mask_copy))
            mask_num=clone_mask_copy.shape[0]
            gaussians_num=gaussians.get_xyz.shape[0]
            differ=gaussians_num-mask_num

            for item in stack:
                world_view_transform, clone_mask, split_mask = item

                if len(clone_vector)==0:
                    clone_vector=clone_mask
                else:
                    clone_vector &= clone_vector
                if len(split_vector)==0:
                    split_vector=split_mask
                else:
                    split_vector &= split_vector
                if len(clone_opt_vector)==0:
                    clone_opt_vector=torch.where(clone_mask, torch.tensor(1).cuda(), torch.tensor(0).cuda())
                    clone_des_vector=torch.where(clone_mask, torch.tensor(1).cuda(), torch.tensor(-1).cuda())
                else:
                    clone_opt_vector+=torch.where(clone_mask, torch.tensor(1).cuda(), torch.tensor(0).cuda())
                    clone_des_vector+=torch.where(clone_mask, torch.tensor(1).cuda(), torch.tensor(-1).cuda())
                if len(split_opt_vector)==0:
                    split_opt_vector=torch.where(split_mask, torch.tensor(1).cuda(), torch.tensor(0).cuda())
                    split_des_vector=torch.where(split_mask, torch.tensor(1).cuda(), torch.tensor(-1).cuda())
                else:
                    split_opt_vector+=torch.where(split_mask, torch.tensor(1).cuda(), torch.tensor(0).cuda())
                    split_des_vector+=torch.where(split_mask, torch.tensor(1).cuda(), torch.tensor(-1).cuda())
                vnum+=1 
            clone_des_vector = torch.where(clone_des_vector >= 0, torch.tensor(True).cuda(), torch.tensor(False).cuda())
            split_des_vector = torch.where(split_des_vector >= 0, torch.tensor(True).cuda(), torch.tensor(False).cuda())
        # A Densification
        if iteration < args.densify_until_iter:

        
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune3(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,clone_des_vector,split_des_vector)

        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:

                clone_opt_vector2=clone_opt_vector%vnum
                split_opt_vector2=split_opt_vector%vnum

                clone_opt_vector=clone_opt_vector2*(clone_opt_vector2-(vnum/2))
                split_opt_vector=split_opt_vector2*(split_opt_vector2-(vnum/2))

                clone_opt_vector1 = torch.where(clone_opt_vector < 0, torch.tensor(-1).cuda(), torch.where(clone_opt_vector > 0, torch.tensor(1).cuda(), clone_opt_vector))
                split_opt_vector1 = torch.where(split_opt_vector < 0, torch.tensor(-1).cuda(), torch.where(split_opt_vector > 0, torch.tensor(1).cuda(), split_opt_vector))                            
                clone_opt_vector=clone_opt_vector1*clone_opt_vector2
                split_opt_vector=split_opt_vector1*split_opt_vector2

                clone_opt_vector[clone_opt_vector > vnum/2] = vnum - clone_opt_vector[clone_opt_vector > vnum/2]
                split_opt_vector[split_opt_vector > vnum/2] = vnum - split_opt_vector[split_opt_vector > vnum/2]

                clone_opt_vector/=vnum
                split_opt_vector/=vnum

                for item in stack:
                    world_view_transform, clone_mask, split_mask = item
                    gaussian_clone=torch.where(clone_mask, torch.tensor(1).cuda(), torch.tensor(-1).cuda())
                    gaussian_split=torch.where(split_mask, torch.tensor(1).cuda(), torch.tensor(-1).cuda())

                    clone_opt_vector3=gaussian_clone*clone_opt_vector
                    split_opt_vector3=gaussian_split*split_opt_vector
                    grad_clone = torch.where(clone_opt_vector3 >= 0, torch.tensor(1).cuda(), clone_opt_vector3)
                    grad_split = torch.where(split_opt_vector3 >= 0, torch.tensor(1).cuda(), split_opt_vector3)

                    grad_vector=grad_clone+grad_split

                    gaussian_dict.add_entry(world_view_transform,grad_vector)


        if iteration > first_opt_iter and first_opt_iter!=0:
            grad_opt_vector=gaussian_dict.get_values(viewpoint_cam.world_view_transform)
            gaussians.mask_gradients2(grad_opt_vector)

        if iteration < args.densify_until_iter:
            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                gaussians.reset_opacity()


        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = False)


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log1 = 0.4 * loss1.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss1": f"{ema_loss_for_log1:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()


            
                    

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":

    # Add directory to sys.path
    dir_name = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(dir_name, './dataset/'))
    sys.path.append(os.path.join(dir_name, '.'))




    # Command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000,30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--lamda_dssim', type=float, default=0.5, help='lambda_dssim')
    parser.add_argument('--low', type=float, default=0.0001, help='rand low')
    parser.add_argument('--high', type=float, default=0.005, help='rand high')
    parser.add_argument('--train_iter', type=int,default=30000, help='train iter')

       


    args = parser.parse_args()
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
