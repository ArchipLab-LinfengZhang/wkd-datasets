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
import os
import copy
from fid_score import *
from inception import*
from test import *

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    opt.continue_train = True
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # load the reference model (trained with 0.25x data)
    teacher_opt = copy.deepcopy(opt)
    teacher_opt.distill = False
    teacher_opt.epoch = opt.teacher_epoch
    teacher_opt.name = opt.teacher_exp_name
    teacher = create_model(teacher_opt)
    teacher.setup(teacher_opt)
    fid_a, fid_b = evaluate(teacher, teacher_opt)
    print("reference model fids", fid_a, fid_b)

    
    # record the weights of referece model
    reference_netG_A = {}
    reference_netG_B = {}

    for name, param in teacher.netG_A.named_parameters():
        reference_netG_A[name] = param.data.clone()
    
    for name, param in teacher.netG_B.named_parameters():
        reference_netG_B[name] = param.data.clone()
    
    # load the teacher model 
    teacher_opt = copy.deepcopy(opt)
    teacher_opt.distill = False
    teacher_opt.epoch = opt.epoch
    teacher_opt.name = opt.name
    teacher = create_model(teacher_opt)
    teacher.setup(teacher_opt)
    fid_a, fid_b = evaluate(teacher, teacher_opt)
    print("teacher model fids", fid_a, fid_b)


    # load the student model
    opt.distill = True
    model = create_model(opt, teacher)  # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.reference_netG_A = reference_netG_A 
    model.reference_netG_B = reference_netG_B

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    fid_a, fid_b = evaluate(model, model.opt)
    print("student fids", fid_a, fid_b)

    best_fid_a_list, best_fid_b_list = [], []
    for index_ in range(5):
        best_fid_a, best_fid_b = 1000, 1000
        print("start pruning ", index_)
        model.prune_generators(ratio = (index_+1)/5)
        print("sparsity ", 1 - model.compute_params())
        
        fid_a, fid_b = evaluate(model, model.opt)
        print("after pruning student fids", fid_a, fid_b)
        best_fid_a = min(best_fid_a, fid_a)
        best_fid_b = min(best_fid_b, fid_b)

        total_iters = 0                # the total number of training iterations
        model.reset_optimizer()
        
        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
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
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)
                #if i > 200:
                 #   break
                iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)
                
                fid_a, fid_b = evaluate(model, model.opt)
                print("after pruning student fids", fid_a, fid_b)
                best_fid_a = min(best_fid_a, fid_a)
                best_fid_b = min(best_fid_b, fid_b)
            
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        best_fid_a_list.append(best_fid_a)
        best_fid_b_list.append(best_fid_b)

    print(best_fid_a_list)
    print(best_fid_b_list)
