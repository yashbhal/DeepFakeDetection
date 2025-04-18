import os
import csv
import torch
import argparse

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *


def main():
    # Running tests
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, default=None, help='Path to save results CSV')
    args, unknown = parser.parse_known_args()

    # Patch sys.argv to remove --results_file so TestOptions().parse() works as before
    import sys
    sys_argv = sys.argv[:]
    if '--results_file' in sys_argv:
        idx = sys_argv.index('--results_file')
        # Remove the option and its value
        sys_argv.pop(idx)
        if idx < len(sys_argv):
            sys_argv.pop(idx)
    sys.argv = sys_argv

    opt = TestOptions().parse(print_options=False)
    model_name = os.path.basename(model_path).replace('.pth', '')
    header = ["testset", "accuracy", "avg precision", "precision", "recall", "f1", "real_acc", "fake_acc", "tn", "fp", "fn", "tp"]
    rows = [["{} model testing on...".format(model_name)], header]

    print("{} model testing on...".format(model_name))

    # Patch: Only use CUDA if available and requested
    use_cuda = torch.cuda.is_available() and hasattr(opt, 'gpu_ids') and len(opt.gpu_ids) > 0
    if use_cuda:
        device = torch.device(f'cuda:{opt.gpu_ids[0]}')
    else:
        device = torch.device('cpu')

    print("\n=== Color Jitter Settings ===")
    print(f"color_jitter_prob: {getattr(opt, 'color_jitter_prob', None)}")
    print(f"brightness_factor: {getattr(opt, 'brightness_factor', None)}")
    print(f"contrast_factor: {getattr(opt, 'contrast_factor', None)}")
    print(f"saturation_factor: {getattr(opt, 'saturation_factor', None)}")
    print(f"hue_factor: {getattr(opt, 'hue_factor', None)}")
    print("===========================\n")

    # Print class-to-index mapping for label verification
    # We'll load the dataset in the same way as in binary_dataset
    from data.datasets import binary_dataset
    test_dataset = binary_dataset(opt, opt.dataroot)
    print("Class to index mapping:", test_dataset.class_to_idx)

    for v_id, val in enumerate(vals):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        opt.no_resize = True    # testing without resizing by default

        model = resnet50(num_classes=1)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict['model'])
        if use_cuda:
            model.cuda()
        else:
            model.cpu()
        model.eval()

        metrics = validate(model, opt, device=device)
        cm = metrics['confusion_matrix']
        # confusion_matrix: [[tn, fp], [fn, tp]]
        tn, fp = cm[0] if len(cm) > 0 else (None, None)
        fn, tp = cm[1] if len(cm) > 1 else (None, None)
        rows.append([
            val,
            metrics['accuracy'],
            metrics['avg_precision'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['real_acc'],
            metrics['fake_acc'],
            tn, fp, fn, tp
        ])
        print(f"({val}) acc: {metrics['accuracy']}; ap: {metrics['avg_precision']}; precision: {metrics['precision']}; recall: {metrics['recall']}; f1: {metrics['f1']}")
        print(f"      real_acc: {metrics['real_acc']}; fake_acc: {metrics['fake_acc']}")
        print(f"      confusion_matrix: {cm}")

    results_file = args.results_file
    results_csv_path = results_file if results_file else 'results/{}.csv'.format(model_name)
    with open(results_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

if __name__ == "__main__":
    main()
