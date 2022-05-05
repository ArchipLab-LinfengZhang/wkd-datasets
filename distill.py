"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import os
# we change the codes in test.py to evaluate fids more conveniently
from test import *

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    opt.distill = True
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)


    # default teacher has the same options except for more ngf (64)
    teacher_opt = copy.copy(opt)
    teacher_opt.ngf = 64
    teacher_opt.distill = False
    teacher_opt.name = opt.teacher_exp_name
    teacher = create_model(teacher_opt)
    teacher.isTrain = False
    teacher.setup(opt)
    # evaluting the fid scoares of the used teacher model
    print("evaluating teacher fids")
    fid_a, fid_b = evaluate(teacher, teacher_opt)
    print("teacher fids ", fid_a, fid_b)
    model = create_model(opt, teacher)  # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # we disable all the visualier in original codes of CycleGAN/Pix2Pix. 
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    total_iters = 0                # the total number of training iterations
    student_fid = [[], []]
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        #visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                #visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print (losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            fid_a, fid_b = evaluate(model, model.opt)
            print("student fids ", fid_a, fid_b)
            student_fid[0].append(fid_a)
            student_fid[1].append(fid_b)
            print("best student fid", min(student_fid[0]), min(student_fid[1]))
            model.save_networks('latest')
            model.save_networks(epoch)
            

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    print("student fids", student_fid)
    print("best student fid", min(student_fid[0]), min(student_fid[1]))
