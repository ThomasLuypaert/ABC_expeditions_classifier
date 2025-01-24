import torch
import torch.nn as nn

from ct_classifier.util import * #SPECIES2CLASS, SPECIES2ORDER, SPECIES2FAMILY, SPECIES2INDEX


def map_indices():

    # Convert the prediction/true labels to their correspondent strings

    # species_string = [SPECIES2INDEX.get(index) for index in species_indices]

    # Convert species strings to their respective class, order, and family strings

    # class_string = [SPECIES2CLASS.get(species) for species in species_string]
    # order_string = [SPECIES2ORDER.get(species) for species in species_string]
    # family_string = [SPECIES2FAMILY.get(species) for species in species_string]

    # Get dictionaries with class/order/family to indices

    class_to_idx = dict([c, idx]  for idx, c in enumerate(set({value: key for key, value in SPECIES2CLASS.items()})))
    order_to_idx = dict([c, idx]  for idx, c in enumerate(set({value: key for key, value in SPECIES2ORDER.items()})))
    family_to_idx = dict([c, idx]  for idx, c in enumerate(set({value: key for key, value in SPECIES2FAMILY.items()})))

    # Convert the class, order, and family strings to indices

    # class_indices = [class_to_idx.get(classes) for classes in class_string]
    # order_indices = [order_to_idx.get(orders) for orders in order_string]
    # family_indices = [family_to_idx.get(families) for families in family_string]

    # Create a link between species indices and the indices of class, order, family

    species_class_indices = [class_to_idx[classes] for classes in [SPECIES2CLASS[value] for key, value in SPECIES2INDEX.items()]]
    species_order_indices = [order_to_idx[orders] for orders in [SPECIES2ORDER[value] for key, value in SPECIES2INDEX.items()]]
    species_family_indices = [family_to_idx[families] for families in [SPECIES2FAMILY[value] for key, value in SPECIES2INDEX.items()]]

    mapped_sp_class_dict = {key: species_class_indices[key] for key in SPECIES2INDEX}
    mapped_sp_order_dict = {key: species_order_indices[key] for key in SPECIES2INDEX}
    mapped_sp_family_dict = {key: species_family_indices[key] for key in SPECIES2INDEX}

    # Make a new dictionary that holds the new class, order, family indices and the corresponding species indices
    unique_values_class = set(mapped_sp_class_dict.values())
    grouped_dict_class = {value: [] for value in unique_values_class}

    for key, value in mapped_sp_class_dict.items():
        grouped_dict_class[value].append(key)

    unique_values_order = set(mapped_sp_order_dict.values())
    grouped_dict_order = {value: [] for value in unique_values_order}

    for key, value in mapped_sp_order_dict.items():
        grouped_dict_order[value].append(key)

    unique_values_family = set(mapped_sp_family_dict.values())
    grouped_dict_family = {value: [] for value in unique_values_family}

    for key, value in mapped_sp_family_dict.items():
        grouped_dict_family[value].append(key)

    # Wrap it all together

    sp_to_group_ind_dict = {"class": mapped_sp_class_dict, 
                            "order": mapped_sp_order_dict,
                            "family": mapped_sp_family_dict}

    aggregation_dict = {"class": grouped_dict_class, 
                  "order": grouped_dict_order, 
                  "family" : grouped_dict_family
                  }

    return sp_to_group_ind_dict, aggregation_dict


# class HierarchicalLoss(nn.Module):

#     def __init__(self):



#     def forward(self, input: Tensor, target: Tensor) -> Tensor:

#         """
#         Compute the hierarchical loss.
#         """

#         F.cross_entropy(
#             input,
#             target
#         )