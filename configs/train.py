'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''

import os
import argparse
import yaml
import glob
from tqdm import trange
import wandb
import random

import torch # this imports pytorch
from torchmetrics import F1Score, Precision, Recall, ConfusionMatrix, Accuracy
import torch.nn as nn # this contains our loss function 
from torch.optim import SGD # this imports the optimizer

# let's import our own classes and functions!
from ct_classifier.util import init_seed, create_confusion_matrix, ann_by_mistake
from ct_classifier.dataset import CTDataset, create_dataloader
from ct_classifier.model import CustomResNet18, CustomResNet50
from ct_classifier.inference import model_inference


def load_model(cfg, model_depth = "ResNet18"):
    '''
        Creates a model instance and loads the latest model state weights.
    '''

    if model_depth == "ResNet18":
        model_instance = CustomResNet18(cfg['num_classes'])  # create an object instance of our CustomResNet18 class
        
    if model_depth == "ResNet50":
        model_instance = CustomResNet50(cfg["num_classes"])  # create an object instance of our CustomResNet50 class     

    # load latest model state
    model_loc = os.path.join(cfg["save_path"], cfg["experiment_name"])
    model_states = glob.glob(os.path.join(model_loc, "*.pt"))
    
    # if len(model_states):
    #     # at least one save state found; get latest
    #     model_epochs = [int(m.replace('model_states/','').replace('.pt','')) for m in model_states]
    #     start_epoch = max(model_epochs)

    #     # load state dict and apply weights to model
    #     print(f'Resuming from epoch {start_epoch}')
    #     state = torch.load(open(f'model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
    #     model_instance.load_state_dict(state['model'])

    # else:
    #     # no save state found; start anew
    print('Starting new model')
    start_epoch = 0

    return model_instance, start_epoch


def save_model(cfg, model, stats, ckpt_name):

    # make sure save directory exists; create if not
    experiment_name = cfg["experiment_name"]
    final_path = os.path.join(cfg["save_path"], experiment_name)

    os.makedirs(final_path, exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(os.path.join(final_path, ckpt_name), 'wb'))
    
    # also save config file if not present
    cfpath = os.path.join(final_path, "config.yaml")
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)

            

def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer



def train(cfg, dataLoader, model, optimizer):
    '''
        Our actual training function.
    '''

    device = cfg['device']

    # put model on device
    model.to(device)
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # loss function
    #  note: if you're doing multi target classification, use nn.BCEWithLogitsLoss() and convert labels to float
    criterion = nn.CrossEntropyLoss()

    # running averages
    loss_total, oa_total = 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)
    
    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    for idx, (data, labels, _) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label

        # put data and labels on device
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total



def validate(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # Initialize torchmetrics
    num_classes = cfg['num_classes']  # Ensure this is defined in your config
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    precision = Precision(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    recall = Recall(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    f1_score = F1Score(task="multiclass", num_classes=num_classes, average="weighted").to(device)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels, _) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            # Update torchmetrics
            accuracy.update(pred_label, labels)
            precision.update(pred_label, labels)
            recall.update(pred_label, labels)
            f1_score.update(pred_label, labels)

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    # Compute metrics at the end of the epoch
    accuracy = accuracy.compute()
    precision = precision.compute()
    recall = recall.compute()
    f1_score = f1_score.compute()

    print(f"accuracy ; {accuracy}")
    print(f"precision ; {precision}")
    print(f"recall ; {recall}")
    print(f"f1_score ; {f1_score}")

    # Reset metrics for the next epoch
    #accuracy.reset()
    #precision.reset()
    #recall.reset()
    #f1_score.reset()

    return loss_total, oa_total, accuracy, precision, recall, f1_score



def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # Initialize weights and biases

    wandb.init(
        
        # set the wandb project where this run will be logged
        project="my-awesome-project",
        entity = "thomas-luypaert-norwegian-university-of-life-sciences",
        name = cfg["experiment_name"],
        
        # track hyperparameters and run metadata
        
        config={
            "learning_rate": cfg["learning_rate"],
            "architecture": "ResNet",
            "dataset": "ABC-expeditions",
            "epochs": cfg["num_epochs"]}
            )
    

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='test')
    dl_all = create_dataloader(cfg, split='none')

    # initialize model
    model, current_epoch = load_model(cfg, model_depth = cfg["model_depth"])

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    best_precision = 0

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val, accuracy, precision, recall, f1_score = validate(cfg, dl_val, model)

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val,
            "accuracy": accuracy, 
            "precision": precision, 
            "recall": recall,
            "f1-score": f1_score
        }

         # Send to wandb
         
        wandb.log({"val_loss": loss_val,
                   "train_loss": loss_train,
                   "val_oa": oa_val,
                   "train_oa": oa_train,
                   "acc": accuracy, 
                   "precision":precision,
                   "recall":recall,
                   "f1-score":f1_score})

        # wandb.save(str()) # To save the checkpoints to wandb

        if precision > best_precision:
            best_precision = precision
            
            save_model(cfg, model, stats, ckpt_name = "best.pt")

        if current_epoch == cfg["num_epochs"]:
            save_model(cfg, model, stats, ckpt_name = "last.pt")

    # Generate a summary_df on the last epoch weights

    checkpoint_location = os.path.join(cfg["save_path"], cfg["experiment_name"])

    best_checkpoint = os.path.join(checkpoint_location, "best.pt")

    df_summary_val = model_inference(config = cfg, 
                                 checkpoint_path=best_checkpoint, 
                                 split = "test", 
                                 type = "summary")
    
    df_summary_train = model_inference(config = cfg,
                                       checkpoint_path=best_checkpoint, 
                                       split = "train", 
                                       type = "summary")
    
    # Create confusion matrix

    dictionary_for_mapping = dl_all.dataset.inv_labels

    conf_mat = create_confusion_matrix(summary_df = df_summary_val, 
                                       config = cfg, 
                                       mapping_dict = dictionary_for_mapping)

    # Create annotations vs mistakes plot

    mistakes_plot = ann_by_mistake(summary_df = df_summary_train)

    # Pipe to Weights and Biases

    wandb.log({ 'chart' : wandb.Image(conf_mat) }, 
              { 'chart' : wandb.Image(mistakes_plot)})

    # That's all, folks!
        


if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
