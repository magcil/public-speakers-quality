import torch
import sys
import math
from copy import deepcopy
import numpy as np


def train_and_validate(model,
                       train_loader,
                       valid_loader,
                       loss_function,
                       optimizer,
                       epochs,
                       validation_epochs=5,
                       early_stopping=False,
                       patience=20):
    """
    Trains the given <model>.
    Then validates every <validation_epochs>.
    Returns: <best_model> containing the model with best parameters.
    """

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min',
                                                           verbose=True)

    # obtain the model's device ID
    device = next(model.parameters()).device

    #print(next(iter(train_loader)))

    EPOCHS = epochs
    VALIDATION_EPOCHS = validation_epochs

    # Store losses, models
    all_train_loss = []
    all_valid_loss = []
    all_metric_training = []
    all_metric_validation = []
    all_valid_comparison_metric = []
    best_model = None
    best_model_epoch = 0

    comparison_metric_max = 1e5
    early_stop_counter = 0

    # Iterate for EPOCHS
    for epoch in range(1, EPOCHS + 1):

        scheduler.step(epoch)
        # ===== Training HERE =====
        train_loss, train_metric = train(
            epoch, train_loader, model,
            loss_function, optimizer)
        # Store statistics for later usage
        all_train_loss.append(train_loss)
        all_metric_training.append(train_metric)

        # ====== VALIDATION HERE ======

        valid_loss, valid_metric, comparison_metric = validate(
            epoch, valid_loader, model, loss_function,
            validation_epochs)

        # Find best model
        if best_model is None:
            # Initialize
            # Store model but on cpu
            best_model = deepcopy(model).to('cpu')
            best_model_epoch = epoch
            # Save new minimum
            comparison_metric_max = comparison_metric
        # New model with lower loss

        elif (comparison_metric < comparison_metric_max - 1e-6):
            # Update loss
            comparison_metric_max = comparison_metric
            # Update best model, store on cpu
            best_model = deepcopy(model).to('cpu')
            best_model_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Store statistics for later usage
        all_valid_loss.append(valid_loss)

        all_valid_comparison_metric.append(comparison_metric)
        # Make sure enough epochs have passed
        if epoch < 4 * VALIDATION_EPOCHS:
            continue

        # Early stopping enabled?
        if early_stopping is False:
            continue
        # If enabled do everything needed
        STOP = True

        # If validation loss is ascending two times in a row exit training
        if early_stop_counter > patience:
            print(f'\nIncreasing loss..')
            print(f'\nResetting model to epoch {best_model_epoch}.')
            # Remove unnessesary model
            model.to('cpu')
            best_model = best_model.to(device)
            # Exit 2 loops at the same time, go to testing
            return best_model, all_train_loss, all_valid_loss, \
                   all_metric_training, all_metric_validation,\
                   all_valid_comparison_metric, epoch


    print(f'\nTraining exited normally at epoch {epoch}.')
    # Remove unnessesary model
    model.to('cpu')
    best_model = best_model.to(device)
    return best_model, all_train_loss, all_valid_loss, \
           all_metric_training, all_metric_validation,\
           all_valid_comparison_metric, epoch


def train(_epoch, dataloader, model, loss_function,
          optimizer):
    # Set model to train mode
    model.train()
    training_loss = 0.0
    correct = 0
    loss_aggregated = 0

    # obtain the model's device ID
    device = next(model.parameters()).device

    # Iterate the batch
    for index, batch in enumerate(dataloader, 1):

        # Split the contents of each batch[i]
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass: y' = model(x)


        # We got a CNN
        # Add a new axis for CNN filter features, [z-axis]
        inputs = inputs[:, np.newaxis, :, :]
        y_pred = model.forward(inputs)

        loss = loss_function(y_pred[:, 0], labels)
        loss_aggregated += loss.item() * inputs.size(0)

        # print(f'\ny_preds={y_pred}')
        # print(f'\nlabels={labels}')
        # Compute loss: L = loss_function(y', y)

        # Backward pass: compute gradient wrt model parameters
        loss.backward()

        # Update weights
        optimizer.step()

        # Add loss to total epoch loss
        training_loss += loss.data.item()

        # print statistics
        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))

    # print statistics
    progress(loss=training_loss / len(dataloader),
             epoch=_epoch,
             batch=index,
             batch_size=dataloader.batch_size,
             dataset_size=len(dataloader.dataset))


    score = loss_aggregated / (len(dataloader) * dataloader.batch_size)
    # Print some stats
    # print(
    #     f'\nTrain loss at epoch {_epoch} : {round(training_loss/len(dataloader), 4)}')
    # Return loss, accuracy
    return training_loss / len(dataloader), score


def validate(_epoch, dataloader, model, loss_function,
             validation_epochs):
    """Validate the model."""

    # Put model to evalutation mode
    model.eval()

    correct = 0
    loss_aggregated = 0
    # obtain the model's device ID
    device = next(model.parameters()).device

    with torch.no_grad():

        pred_all = []
        actual_labels = []
        for index, batch in enumerate(dataloader, 1):

            # Get the sample
            inputs, labels = batch

            # Transfer to device
            inputs = inputs.to(device)
            labels = labels.type('torch.LongTensor').to(device)


            # We got CNN
            # Add a new axis for CNN filter features, [z-axis]
            inputs = inputs[:, np.newaxis, :, :]
            y_pred = model.forward(inputs)

            loss = loss_function(y_pred[:, 0], labels)

            loss_aggregated += loss.item() * inputs.size(0)

        val_loss = loss_aggregated / (len(dataloader) * dataloader.batch_size)


        score = val_loss
        comparison_metric = score

        if _epoch % validation_epochs == 0:
            # Print some stats
            print('\nValidation results for epoch {}:'.format(_epoch))

            print('    --> MAE: {}'.format(round(score, 4)))

    return val_loss, score, comparison_metric



def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()
