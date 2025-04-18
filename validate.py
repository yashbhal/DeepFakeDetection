import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from options.test_options import TestOptions
from data import create_dataloader
from tqdm import tqdm


def validate(model, opt, device=None):
    data_loader = create_dataloader(opt)
    total = len(data_loader.dataset) if hasattr(data_loader, 'dataset') else None
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        y_true, y_pred = [], []
        print(f"Evaluating on {len(data_loader)} batches...")
        for i, (img, label) in enumerate(tqdm(data_loader, desc='Eval', unit='batch')):
            img = img.to(device)
            label = label.to(device)
            pred = model(img).sigmoid().flatten().tolist()
            y_pred.extend(pred)
            y_true.extend(label.flatten().tolist())
            if (i+1) % 10 == 0:
                print(f"Processed {i+1} batches, {len(y_true)} samples")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_bin = (y_pred > 0.5).astype(int)

    # Main metrics
    acc = accuracy_score(y_true, y_pred_bin)
    ap = average_precision_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred_bin, zero_division=0)
    recall = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_bin)
    # Per-class accuracy
    real_acc = accuracy_score(y_true[y_true==0], y_pred_bin[y_true==0]) if np.any(y_true==0) else float('nan')
    fake_acc = accuracy_score(y_true[y_true==1], y_pred_bin[y_true==1]) if np.any(y_true==1) else float('nan')

    metrics = {
        'accuracy': acc,
        'avg_precision': ap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'real_acc': real_acc,
        'fake_acc': fake_acc,
        'confusion_matrix': cm.tolist(),
    }
    return metrics


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    if use_cuda:
        model.cuda()
    model.eval()

    metrics = validate(model, opt, device=device)

    print("accuracy:", metrics['accuracy'])
    print("average precision:", metrics['avg_precision'])
    print("precision:", metrics['precision'])
    print("recall:", metrics['recall'])
    print("f1:", metrics['f1'])
    print("accuracy of real images:", metrics['real_acc'])
    print("accuracy of fake images:", metrics['fake_acc'])
    print("confusion matrix:\n", metrics['confusion_matrix'])
