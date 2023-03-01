import os
import numpy as np
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import EPOCHS, VARIABLES_FOLDER, BATCH_SIZE
from cnn import CNN1
from training import train_and_validate
import time
import pickle
from utils import split_mel_spec
from sklearn.model_selection import GroupShuffleSplit
import numpy
from sklearn.metrics import mean_absolute_error
import argparse

def aggregated_valid_seg(x_test, y_test, groups_test, best_model):
    y_pred = []
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device ='cpu'
    #x_test = x_test.to(device)

    best_model.to(device)
    best_model.eval()
    with torch.no_grad():
        for sample in x_test:
            tensor_x = torch.Tensor(sample)
            tensor_x.to(device)
            input = tensor_x[np.newaxis, np.newaxis, :, :].to(device)
            y_pred.append(best_model.forward(input))

    averaged_preds, averaged_gr_truths = [], []
    for group_id in set(groups_test):
        ground_truths_of_group = [i for j, i in enumerate(y_test) if groups_test[j] == group_id]
        preds_of_group = [i.numpy() for j, i in enumerate(y_pred) if groups_test[j] == group_id]
        averaged_preds.append(round(numpy.average(preds_of_group), 2))
        averaged_gr_truths.append(ground_truths_of_group[0])
    error = mean_absolute_error(averaged_gr_truths, averaged_preds)
    return error

def train(X, y, grps, ofile=None, random_state=42):

    """Train a given model on a given dataset"""
    # Check that folders exist
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True


    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=random_state)
    split = splitter.split(X, groups=grps)
    train_inds, test_inds = next(split)
    x_train = X[train_inds, :, :]
    y_train = y[train_inds]
    x_test = X[test_inds, :, :]
    y_test = y[test_inds]
    groups_test = [grps[index] for index in test_inds]

    tensor_x = torch.Tensor(x_train)  # transform to torch tensor
    tensor_y = torch.Tensor(y_train)

    train_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              num_workers=4, drop_last=True, shuffle=True)  # create your dataloader

    tensor_x = torch.Tensor(x_test)  # transform to torch tensor
    tensor_y = torch.Tensor(y_test)

    valid_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                              num_workers=4, drop_last=True, shuffle=True)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")
    # Create model and send to device


    height = x_train[0].shape[0]
    width = x_train[0].shape[1]
    print(height, width)
    spec_size = (X[0].shape[0], X[0].shape[1])
    print(spec_size)
    model = CNN1(height=height, width=width,
                 output_dim=1, spec_size=spec_size)

    model.to(device)
    # Add max_seq_length to model
    max_seq_length = X[0].shape[1]
    model.max_sequence_length = max_seq_length
    print('Model parameters:{}'.format(sum(p.numel()
                                           for p in model.parameters()
                                           if p.requires_grad)))

    ##################################
    # TRAINING PIPELINE
    ##################################
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=0.002,
                                  weight_decay=.02)


    loss_function = torch.nn.L1Loss()

    best_model, train_losses, valid_losses, \
    train_metric, val_metric, \
    val_comparison_metric, _epochs = train_and_validate(
        model=model, train_loader=train_loader,
        valid_loader=valid_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        epochs=EPOCHS,
        validation_epochs=5,
        early_stopping=True)
    timestamp = time.ctime()
    timestamp = timestamp.replace(" ", "_")

    print('All validation errors: {} \n'.format(val_comparison_metric))
    best_index = val_comparison_metric.index(min(val_comparison_metric))
    best_model_error = val_comparison_metric[best_index]
    print('Best model\'s validation error: {}'.format(best_model_error))
    best_model_loss = valid_losses[best_index]
    print('Best model\'s validation loss: {}'.format(best_model_loss))



    error = aggregated_valid_seg(x_test, y_test, groups_test, best_model)
    print('***********************************************')
    print("The error on aggregated segments is: ", error)

    if ofile is None:
        ofile = f"{best_model.__class__.__name__}_{_epochs}_{timestamp}.pt"
    else:
        ofile = str(ofile)
        if '.pt' not in ofile or '.pkl' not in ofile:
            ofile = ''.join([ofile, '.pt'])
    if not os.path.exists(VARIABLES_FOLDER):
        os.makedirs(VARIABLES_FOLDER)
    modelname = os.path.join(
        VARIABLES_FOLDER, ofile)

    best_model = best_model.to("cpu")
    print(f"\nSaving model to: {modelname}\n")

    # Save model for later use
    model_params = {
        "height": height, "width": width,
        "spec_size": spec_size,
        "max_sequence_length": max_seq_length,
        "type": best_model.type, "state_dict": best_model.state_dict()
    }


    model_params["validation_error"] = best_model_error
    with open(modelname, "wb") as output_file:
        pickle.dump(model_params, output_file)

    return error

def train_cross_validation(gt_file, f_dir, task, ofile):
    K=20

    with open(gt_file) as fp:
        ground_truth = json.load(fp)


    X, y, grps = [], [], []
    counter = 0
    for ig, g in enumerate(ground_truth):
        npy_melgram = os.path.join(f_dir, ground_truth[ig]['name'].replace('.wav', '')) + '_melgram.npy'
        gt = ground_truth[ig][task]['mean']
        if gt is not None:
            fv = np.load(npy_melgram)
            counter += 1
            segments = split_mel_spec(fv)
            X += segments
            y += [gt] * len(segments)
            grps += [counter] * len(segments)
    X = np.stack(X, axis=0)
    y = np.array(y)
    errors = []
    for i in range(K):
        errors.append(train(X, y, grps, ofile, i))
    print('****************************************')
    mean_error = round(numpy.average(errors), 2)
    print("The mean cross validated error on aggregated segments is: ", mean_error)

if __name__ == '__main__':

    # Read arguments -- a list of folders
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", "--ground_truth", type=str, nargs="?", required=True,
        help="Ground truth and metadata json file")
    parser.add_argument(
        "-f", "--features_folder", type=str, nargs="?", required=True,
        help="Folder where the features are stored in npy files")
    parser.add_argument(
        "-t", "--task_name", type=str, nargs="?", required=True,
        help="Task to be solved")
    parser.add_argument('-o', '--ofile', required=False, default=None,
                        type=str, help='Model name.')
    args = parser.parse_args()

    # Get argument
    flags = parser.parse_args()
    gt_file = flags.ground_truth
    f_dir = flags.features_folder
    task = flags.task_name
    ofile = flags.ofile

    train_cross_validation(gt_file, f_dir, task, ofile)