
import torch
import pandas as pd

import yaml

from torch.nn import Softmax

from ct_classifier.train import create_dataloader
from ct_classifier.dataset import CTDataset
from ct_classifier.model import CustomResNet18


def model_inference(config,checkpoint_path ):

    # 0. Parameters

    cfg = yaml.safe_load(open(config, 'r'))

    # 1. Loading the model

    device = cfg['device']

    model = CustomResNet18(cfg['num_classes'])

    # 2. Grab the latest model (if pre-trained model exists)

    state = torch.load(open(checkpoint_path, 'rb'), map_location=device)
    model.load_state_dict(state['model'])
    model.to(device)
    model.eval()

    # 3. Prepare the data loader

    dl_val = create_dataloader(cfg, split='test')

    # 4. Apply the latest model on the validation data and save metrics

    prediction_labels = []
    prediction_confs = []
    true_labels = []
    file_list = []

    for idx, (data, labels, file_names) in enumerate(dl_val):

        data = data.to(device)

        # Predict labels on validation data

        s = Softmax(dim=1)
        
        with torch.no_grad():
             prediction = model(data)

        prediction = s(prediction)
        pred_label = torch.argmax(prediction, dim=1)
        pred_conf = prediction.max(dim = 1).values

        prediction_labels.append(pred_label.cpu().numpy())
        prediction_confs.append(pred_conf.cpu().numpy())

        # Get ground-truth labels 

        true_labels.append(labels)
        file_list.append(file_names)

    print(len(prediction_labels))
    print(len(prediction_confs))
    print(len(true_labels))
    print(len(file_names))
    
    prediction_df = {'pred_labels': prediction_labels,
                     "pred_confs" : prediction_confs, 
                     "true_labels": true_labels, 
                     "file_names": file_names}
    
    prediction_df = pd.DataFrame(prediction_df)

    return prediction_df




