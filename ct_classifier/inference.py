
import torch
import pandas as pd
import numpy as np

import yaml

from torch.nn import Softmax

from ct_classifier.dataset import CTDataset, create_dataloader
from ct_classifier.model import CustomResNet18


def model_inference(config,
                    checkpoint_path, 
                    split = "test", 
                    type = "summary"):

    # 0. Parameters

    #cfg = yaml.safe_load(open(config, 'r'))

    # 1. Loading the model

    device = config['device']

    model = CustomResNet18(config['num_classes'])

    # 2. Grab the latest model (if pre-trained model exists)

    state = torch.load(open(checkpoint_path, 'rb'), map_location=device)
    model.load_state_dict(state['model'])
    model.to(device)
    model.eval()

    # 3. Prepare the data loader

    if split == "test":
        dl = create_dataloader(config, split='test')

    if split == "train":
        dl = create_dataloader(config, split = "train")

    # 4. Apply the latest model on the validation data and save metrics

    prediction_labels = []
    prediction_confs = []
    true_labels = []
    true_labels_conv = []
    file_list = []
    prediction_labels_conv = []

    prediction_df_raw = []

    label_dict = dl.dataset.inv_labels

    for idx, (data, labels, file_names) in enumerate(dl):

        data = data.to(device)

        # Predict labels on validation data

        s = Softmax(dim=1)
        
        with torch.no_grad():
            prediction = model(data)

        # Grab raw prediction scores per sample

        prediction_raw = prediction # no softmax applied
        prediction_raw = prediction_raw.cpu().numpy()

        colnames = range(prediction_raw.shape[1])
        colnames = [label_dict[x] for x in colnames]

        prediction_raw = pd.DataFrame(prediction_raw, columns=colnames)

        # Grab max softmax prediction score per sample        
        prediction = s(prediction)
        
        # Grab prediction labels and confidences    
        pred_label = torch.argmax(prediction, dim=1)
        pred_label_conv = [label_dict[int(x.item())] for x in pred_label]
        pred_conf = prediction.max(dim = 1).values 

        prediction_labels.append(pred_label.cpu().numpy())
        prediction_labels_conv.append(pred_label_conv)
        prediction_confs.append(pred_conf.cpu().numpy())

        # Get ground-truth labels 

        labels_conv = [label_dict[int(x.item())] for x in labels]

        true_labels.append(labels)
        true_labels_conv.append(labels_conv)

        file_list.append(file_names)

        # Create data frame 1 (raw prediction scores per frame)

        prediction_raw.insert(0, "true_label", np.array(labels_conv))
        prediction_raw.insert(0, 'file_name', file_names)

        prediction_df_raw.append(prediction_raw)

    prediction_df_raw = pd.concat(prediction_df_raw, ignore_index=True)
    prediction_df = {'pred_labels': np.hstack(prediction_labels),
                     "pred_labels_full": np.hstack(prediction_labels_conv),
                     "pred_confs" : np.hstack(prediction_confs), 
                     "true_labels": np.hstack(true_labels), 
                     "true_labels_full": np.hstack(true_labels_conv),
                     "file_names": np.hstack(file_list)}
    
    prediction_df = pd.DataFrame(prediction_df)

    if type == "raw":
        return prediction_df_raw

    if type == "summary":
        return prediction_df



