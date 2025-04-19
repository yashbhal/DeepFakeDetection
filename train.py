import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model weights for fine-tuning')
    parser.add_argument('--finetuned_model_path', type=str, default=None, help='Where to save the best fine-tuned model')
    parser.add_argument('--niter', type=int, default=50, help='Number of epochs to train')
    args, unknown = parser.parse_known_args()

    # Remove these args from sys.argv so TrainOptions().parse() works as before
    import sys
    sys_argv = sys.argv[:]
    for arg in ['--pretrained_path', '--finetuned_model_path', '--niter']:
        if arg in sys_argv:
            idx = sys_argv.index(arg)
            sys_argv.pop(idx)
            if idx < len(sys_argv):
                sys_argv.pop(idx)
    sys.argv = sys_argv

    opt = TrainOptions().parse()
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    # --- Force both classes for training ---
    opt.classes = ['real', 'deepfake']
    print(f"Using classes: {opt.classes} in {opt.dataroot}")
    val_opt = get_val_opt()

    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = Trainer(opt)
    # Load pretrained weights if specified
    if args.pretrained_path is not None:
        print(f"Loading pretrained weights from {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path, map_location='cpu')
        if 'model' in state_dict:
            model.model.load_state_dict(state_dict['model'], strict=False)
        else:
            model.model.load_state_dict(state_dict, strict=False)

    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    best_val_metric = -float('inf')
    for epoch in range(args.niter):
        epoch_start_time = time.time()
        print(f"\n======== Epoch {epoch+1}/{args.niter} ========")
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.name, epoch, model.total_steps))
                model.save_networks('latest')

        # Log epoch timing
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds ({epoch_time/60:.2f} min)")
        if epoch > 0:
            avg_time = (time.time() - epoch_start_time) / (epoch+1)
            eta = avg_time * (args.niter - (epoch+1))
            print(f"Estimated time remaining: {eta/60:.2f} min ({eta/3600:.2f} hr)")
        else:
            first_epoch_time = epoch_start_time

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Validation
        model.eval()
        val_metrics = validate(model.model, val_opt)
        acc = val_metrics['accuracy']
        ap = val_metrics['avg_precision']
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print(f"(Val @ epoch {epoch}) acc: {acc}; ap: {ap}")

        if acc > best_val_metric:
            best_val_metric = acc
            save_path = args.finetuned_model_path if args.finetuned_model_path else os.path.join(opt.checkpoints_dir, opt.name, f'finetuned_colorjitter_{args.niter}epochs.pth')
            print(f"Saving best fine-tuned model to {save_path}")
            torch.save({'model': model.model.state_dict()}, save_path)

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()
