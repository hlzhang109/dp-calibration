from pydoc import tempfilepager
from netcal.scaling import TemperatureScaling
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram

import matplotlib.pyplot as plt
import numpy as np
import torch
from os.path import join
from temperature_scaling.temp_scalor import DPTemperatureScaling
import json
from transformers import HfArgumentParser, set_seed
from torch.utils.data import DataLoader, TensorDataset
import diffprivlib.models as dp
from sklearn.metrics import accuracy_score
import os
from torch.nn import functional as F
from calibration.common import ECELoss

def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def main(model, max_iter=30, num_class=3, epsilon=8.0, train_dataset='mnli', test_dataset='rte', device='cuda', dp_type = 'sgd', recal_type = 'temp', res_folder = 'ablation'):
    assert train_dataset == test_dataset
    PREFIX = join(res_folder, model, test_dataset) 
    LOGITS_FILE = f"valid-{test_dataset}--1--1.npy"
    PATH = join(PREFIX, LOGITS_FILE)
    LABEL_FILE = f"valid-y-{test_dataset}--1--1.npy"
    LABEL_PATH =  join(PREFIX, LABEL_FILE) 
    #MODEL_PREFIX = join(res_folder, model, "pytorch_model.bin") 

    logits = np.load(PATH)
    y = np.load(LABEL_PATH, allow_pickle=True)
    y_pred = logits.argmax(1)
    logits = torch.tensor(logits).to(device)

    LOGITS_FILE = f"test-{test_dataset}--1--1.npy" 
    PATH =  join(PREFIX, LOGITS_FILE)
    LABEL_FILE = f"test-y-{test_dataset}--1--1.npy"
    LABEL_PATH =  join(PREFIX, LABEL_FILE)

    test_logits = np.load(PATH)
    test_logits = torch.tensor(test_logits).to(device)
    test_y = np.load(LABEL_PATH, allow_pickle=True)
    data = {'train': (logits, torch.tensor(y)), 'test': (test_logits, torch.tensor(test_y))}
    
    print(f"Valid size: {logits.shape}, Test size: {test_logits.shape}")
    print(f"Train: {train_dataset}, Test: {test_dataset}, ID Valid Accuracy: {accuracy_score(y, logits.argmax(-1).cpu().numpy())}, ID test Accuracy: {accuracy_score(test_y, test_logits.argmax(-1).cpu().numpy())}")

    confidences = torch.nn.functional.softmax(torch.tensor(logits), dim=1)
    # NOTE just keep the confidences for class 1 since there is no distribution shift
    
    pos_confidences = confidences[:, 1] if train_dataset != 'mnli' else confidences
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    ground_truth = y.reshape(-1,1)
    pos_confidences = pos_confidences.cpu().numpy()
    temperature = TemperatureScaling()
    temperature.fit(pos_confidences, ground_truth)
    netcal_calibrated = temperature.transform(pos_confidences)
    
    dp_model = DPTemperatureScaling(max_iter, data, num_class=num_class, recal_type=recal_type, dp_type=dp_type) # confidences
    dp_model.set_private(logits, y, epsilon=epsilon)
    dp_model.set_temperature()

    test_confidences = torch.nn.functional.softmax(test_logits, dim=1).cpu().numpy()
    #netcal_calibrated = temperature.transform(test_confidences)
    dp_calibrated = dp_model.forward(test_logits) 
    np.save(join(PREFIX, "dp_calibrated_logits.npy"), dp_calibrated.detach().cpu().numpy())
    
    netcal_calibrated = test_logits * torch.tensor(temperature.temperature).to(device)
    n_bins = 10
    ece = ECELoss(n_bins=n_bins)
    test_y = torch.tensor(test_y).to(device)
    uncalibrated_score = ece(test_logits.softmax(-1), test_y)
    if not torch.is_tensor(netcal_calibrated):
        netcal_calibrated = torch.tensor(netcal_calibrated).to(device)
    netcal_calibrated_score = ece(netcal_calibrated.softmax(-1), test_y)
    dp_calibrated_score = ece(dp_calibrated.softmax(-1), test_y)
    
    n_bins = 10
    ece = ECE(n_bins)
    netcal_score = ece.measure(pos_confidences, ground_truth)
    netcal_uncalibrated_score = ece.measure(test_confidences[:, 1], test_y.cpu().numpy())
    print("Netcal for norecal, dp", netcal_uncalibrated_score, netcal_score)

    #print("ECE loss on test set: ", _ECELoss(10)(dp_calibrated, torch.tensor(y).to(device)))
    #dp_calibrated = torch.nn.functional.softmax(dp_calibrated, dim=1) 

    # if train_dataset == 'mnli':
    #     dp_calibrated = dp_calibrated
    #     pos_confidences = test_confidences
    # elif train_dataset in ['qnli', 'qqp', 'sst-2']:
    #     dp_calibrated = dp_calibrated[:, 1] # NOTE keep class one confidences
    #     pos_confidences = test_confidences[:, 1] 

    # ground_truth = test_y.reshape(-1,1)
    # n_bins = 10
    # ece = ECE(n_bins)
    # uncalibrated_score = ece.measure(pos_confidences, ground_truth)
    # netcal_calibrated_score = ece.measure(netcal_calibrated, ground_truth)
    # if torch.is_tensor(dp_calibrated):
    #     dp_calibrated = dp_calibrated.detach().cpu().numpy()
    # dp_calibrated_score = ece.measure(dp_calibrated, ground_truth)

    # model_name = model
    # diagram = ReliabilityDiagram(n_bins)
    # fig = diagram.plot(pos_confidences, ground_truth, title_suffix="No-Recal Confidence of {} on {}".format(model_name, test_dataset))  
    # CALIBRATION_PREFIX = join("calibration_outputs", model, test_dataset)
    # plt.savefig(join(PREFIX, "reliability_diagram.png"))
    
    # diagram = ReliabilityDiagram(n_bins)
    # if torch.is_tensor(dp_calibrated):
    #     dp_calibrated = dp_calibrated.detach().cpu().numpy()
    # cal_fig = diagram.plot(dp_calibrated, ground_truth, title_suffix="DP-Recal of {} on {}".format(model_name, test_dataset)) 
    # plt.savefig(join(PREFIX, "dp-recal_diagram.png"))

    # diagram = ReliabilityDiagram(n_bins)
    # cal_fig = diagram.plot(netcal_calibrated, ground_truth, title_suffix="NonDP-Recal of {} on {}".format(model_name, test_dataset)) 
    # plt.savefig(join(PREFIX, "netcal-recal_diagram.png"))

    return dp_model, temperature.temperature, uncalibrated_score, netcal_calibrated_score, dp_calibrated_score

SST2_label_map = {'negative':0, 'positive':1}
COLA_label_map = {'unacceptable':0, 'acceptable':1}

WNLI_label_map = {'entailment':1, 'not_entailment': 0} # 1 (entailment), 0 (not_entailment)
QQP_label_map = {'duplicate':1, 'not_duplicate': 0} # 1 (entailment), 0 (not_entailment)
MRPC_label_map = {'equivalent':1, 'not_equivalent': 0} # 1 (entailment), 0 (not_entailment)

RTE_label_map =  {'entailment':0, 'not_entailment': 1} # 1 -- equivalent, 0 -- not equivalent.
QNLI_label_map = {'entailment':0, 'not_entailment': 1} # 1 (not_entailment), 0 (entailment), label_list =  ["entailment", "not_entailment"]
HANS_label_map = {'entailment':0, 'not_entailment': 1} # 1 (not_entailment), 0 (entailment), label_list =  ["entailment", "not_entailment"]
SCITAIL_label_map = {'entailment':0, 'not_entailment': 1} # 1 (not_entailment), 0 (entailment), label_list =  ["entailment", "not_entailment"]

MNLI_label_map = {'contradiction':0, 'neutral':1, 'entailment': 2} # NOTE after swapped 

def label_mapping(confidences, train_dataset, test_dataset): # logits
    #confidences = torch.nn.functional.softmax(logits.clone().detach(), dim=1) # torch.tensor(logits)
    if train_dataset == 'mnli':  # {'contradiction':0, 'neutral':1, 'entailment': 2} 
        entail_prob, not_entail_prob = confidences[:, 2], confidences[:, 0] + confidences[:, 1]
    elif train_dataset == 'qqp': # {'duplicate':1, 'not_duplicate': 0}
        entail_prob, not_entail_prob = confidences[:, 1], confidences[:, 0]
    elif train_dataset == 'qnli': # {'entailment':0, 'not_entailment': 1}
        entail_prob, not_entail_prob = confidences[:, 0], confidences[:, 1]

    if (test_dataset.lower() in ['wnli', 'qqp',  'mrpc']): # {'entailment':1, 'not_entailment': 0}
        confidences = torch.stack([not_entail_prob, entail_prob], dim=-1)
    elif (test_dataset.lower() in ['rte', 'qnli', 'scitail', 'hans', 'sst-2']): # 'hans', # {'entailment':0, 'not_entailment': 1}
        confidences = torch.stack([entail_prob, not_entail_prob], dim=-1)
    pos_confidences = confidences[:, 1] # NOTE keep class one confidences
    y_pred = confidences.argmax(1)
    return confidences, y_pred
    
def ood_main(model, dp_model, netcal_temp, train_dataset='mnli', test_dataset='rte', device='cuda', res_folder = 'ablation'):
    assert train_dataset != test_dataset
    PREFIX = join(res_folder, model, test_dataset)  
    LOGITS_FILE = "valid-{}--1--1.npy".format(test_dataset) 
    PATH = join(PREFIX, LOGITS_FILE)
    LABEL_FILE = "valid-y-{}--1--1.npy".format(test_dataset) 

    LABEL_PATH = join(PREFIX, LABEL_FILE)
    logits = np.load(PATH)
    y = np.load(LABEL_PATH, allow_pickle=True)

    logits = torch.tensor(logits).to(device)
    logits_dataset = TensorDataset(logits)

    dp_calibrated, y_pred = label_mapping(dp_model.forward(logits).softmax(-1), train_dataset, test_dataset)
    netcal_calibrated, y_pred = label_mapping((logits * torch.tensor(netcal_temp).to(device)).softmax(-1), train_dataset, test_dataset)

    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    print(f"Train: {train_dataset}, Test: {test_dataset}, OOD Accuracy: {accuracy_score(y, y_pred)}")

    # if test_dataset.lower() in ['wnli', 'qqp', 'mrpc']: # {'entailment':1, 'not_entailment': 0}
    #     if train_dataset == 'mnli':  # {'contradiction':0, 'neutral':1, 'entailment': 2} 
#         confidences = torch.stack([confidences[:, 0] + confidences[:, 1], confidences[:, 2]], dim=-1)
    #     y_pred = confidences.argmax(1)
    #     pos_confidences = confidences[:, 1]
    # elif test_dataset.lower() in ['rte', 'qnli', 'hans', 'scitail', 'cola', 'sst-2']: # {'entailment':0, 'not_entailment': 1}
    #     if train_dataset == 'mnli':
    #         confidences = torch.stack([confidences[:, 2], confidences[:, 0] + confidences[:, 1]], dim=-1)
    #     y_pred = confidences.argmax(1)
    #     pos_confidences = confidences[:, 1]
    # else:
    #     if train_dataset == 'mnli':
    #         pos_confidences = confidences
    #     elif train_dataset in ['qnli']:
    #         pos_confidences = confidences[:, 0]
    #     elif train_dataset in ['qqp', 'sst-2']:
    #         pos_confidences = confidences[:, 1]

    #ground_truth = y.reshape(-1,1)
    #pos_confidences = pos_confidences.cpu().numpy()
    #netcal_calibrated = pos_confidences * netcal_temp 
    #netcal_calibrated = confidences * torch.tensor(netcal_temp).to(device)

    # dp_calibrated = dp_model.forward(logits) 
    # np.save(join(PREFIX, "dp_calibrated_logits.npy"), dp_calibrated.detach().cpu().numpy())
    # dp_calibrated = torch.nn.functional.softmax(dp_calibrated.clone().detach(), dim=1) # torch.tensor(dp_calibrated)

    n_bins = 10
    ece = ECELoss(n_bins=n_bins)
    y = torch.tensor(y).to(device)
    uncalibrated_score = ece(label_mapping(logits.softmax(-1), train_dataset, test_dataset)[0], y)
    netcal_calibrated_score = ece(torch.tensor(netcal_calibrated).to(device), y)
    dp_calibrated_score = ece(dp_calibrated, y)

    # n_bins = 10
    # ece = ECE(n_bins)
    # netcal_score = ece.measure(pos_confidences, ground_truth)
    # netcal_uncalibrated_score = ece.measure(test_confidences[:, 1], test_y.cpu().numpy())
    # print("Netcal for norecal, dp", netcal_uncalibrated_score, netcal_score)

    # if train_dataset != test_dataset and (test_dataset.lower() in ['wnli', 'qqp', 'mrpc']):
    #     if train_dataset == 'mnli':
    #         dp_calibrated = torch.stack([dp_calibrated[:, 0] + dp_calibrated[:, 1], dp_calibrated[:, 2]], dim=-1)
    #     dp_calibrated = dp_calibrated[:, 1]
    # elif train_dataset != test_dataset and (test_dataset.lower() in ['rte', 'qnli', 'hans', 'scitail', 'cola', 'sst-2']):
    #     if train_dataset == 'mnli':
    #         dp_calibrated = torch.stack([dp_calibrated[:, 2], dp_calibrated[:, 0] + dp_calibrated[:, 1]], dim=-1)
    #     dp_calibrated = dp_calibrated[:, 0]

    # n_bins = 10
    # ece = ECE(n_bins)
    # uncalibrated_score = ece.measure(pos_confidences, ground_truth)
    # netcal_calibrated_score = ece.measure(netcal_calibrated, ground_truth)
    # dp_calibrated_score = ece.measure(dp_calibrated.detach().cpu().numpy(), ground_truth)

    # model_name = model 
    # diagram = ReliabilityDiagram(n_bins)
    # fig = diagram.plot(pos_confidences, ground_truth, title_suffix="No-Recal Confidence of {} on {}".format(model_name, test_dataset)) 
    # CALIBRATION_PREFIX = join("calibration_outputs", model, test_dataset)
    # plt.savefig(join(PREFIX, "reliability_diagram.png"))
    
    # diagram = ReliabilityDiagram(n_bins)
    # cal_fig = diagram.plot(dp_calibrated.detach().cpu().numpy(), ground_truth, title_suffix="DP-Recal of {} on {}".format(model_name, test_dataset))  
    # plt.savefig(join(PREFIX, "dp-recal_diagram.png"))

    # diagram = ReliabilityDiagram(n_bins)
    # cal_fig = diagram.plot(netcal_calibrated, ground_truth, title_suffix="NonDP-Recal of {} on {}".format(model_name, test_dataset))  
    # plt.savefig(join(PREFIX, "netcal-recal_diagram.png"))

    return uncalibrated_score, netcal_calibrated_score, dp_calibrated_score

if __name__ == '__main__':
    test_ls = {
                "mnli": ["mnli", "hans", "scitail", "rte", "wnli"], # "qnli",   # "mrpc",  "qqp",
                "qnli": ["qnli", "hans", "scitail", "rte",  "wnli"],  # "mrpc", "qqp",
                "qqp": ["qqp",  "mrpc",],  # "rte", "qnli", "wnli",  "hans", "scitail"
                "sst-2": ["sst-2"]
              }
    num_class_ls = {"mnli": 3, "qnli": 2, "qqp": 2, "sst-2": 2}
    num_repeats = 1
    seed = 42
    seeds = [i for i in range(seed, num_repeats+seed)]
    res_folder = 'extended' # "nondp" # _full' #'ablation'
    #models = os.listdir(res_folder)
    models = []
    #models.remove('logs') #["QNLI_full_eps8_norm0.1_dp_epoch6_lr1e-3"]  
    # models = ['MNLI_full_eps8_norm0.1_dp_epoch18_lr5e-4'] # , 'MNLI_full_global1000_eps8_norm0.1_dp_epoch18_lr5e-4', 'MNLI_full_sgld_eps8_norm0.1_dp_epoch18_lr5e-4_testlatency']
    # models += ['QNLI_full_eps8_norm0.1_dp_epoch6_lr1e-3',] # 'QNLI_full_global1000_eps8_norm0.1_dp_epoch6_lr1e-3', 'QNLI_full_sgld_eps8_norm0.1_dp_epoch6_lr1e-3_testlatency']
    # models += ['QQP_full_eps8_norm0.1_dp_epoch18_lr5e-4',]# 'QQP_full_global1000_eps8_norm0.1_dp_epoch18_lr5e-4', 'QQP_full_sgld_eps8_norm0.1_dp_epoch18_lr5e-4_testlatency']
    # models += ['SST-2_full_eps8_norm0.1_dp_epoch8_lr1e-3',] # 'SST-2_full_global1000_eps8_norm0.1_dp_epoch8_lr1e-3', 'SST-2_full_sgld_eps8_norm0.1_dp_epoch8_lr1e-3_testlatency']
    #models += ['MNLI_full_eps3_norm0.1_dp_lr1e-4', 'MNLI_full_eps8_norm0.1_dp_lr1e-4', 'MNLI_full_eps11_norm0.1_dp_lr1e-4', 'MNLI_full_eps25_norm0.1_dp_lr1e-4']
    #NOTE global updated
    models += ["MNLI_full_global100_eps8_norm0.1_dp_epoch10_lr5e-4"] # , "QNLI_full_global100_eps8_norm0.1_dp_epoch6_lr1e-3", # "QQP_full_global100_eps8_norm0.1_dp_epoch18_lr5e-4"] # "QNLI_full_global100_eps8_norm0.1_dp_epoch6_lr1e-3"] # ,] # 
    #             "QQP_full_global100_eps8_norm0.1_dp_epoch18_lr5e-4", "SST-2_full_global100_eps8_norm0.1_dp_epoch8_lr1e-3", ] # "QNLI_full_global100_eps8_norm0.1_dp_epoch6_lr1e-3"]
    # NOTE norm ablation
    #models += ["MNLI_full_eps8_norm0.1_dp_epoch18_lr5e-4", "MNLI_full_eps8_norm1_dp_epoch18_lr5e-4", "MNLI_full_eps8_norm10_dp_epoch18_lr5e-4"]
    #NOTE eps ablation
    #models += ["MNLI_full_eps3_norm0.1_dp_epoch18_lr5e-4", "MNLI_full_eps8_norm0.1_dp_epoch18_lr5e-4", "MNLI_full_eps11_norm0.1_dp_epoch18_lr5e-4", "MNLI_full_eps25_norm0.1_dp_epoch18_lr5e-4"]
    #NOTE global norm ablation
    # models += ["MNLI_full_global100_eps8_norm0.1_dp_epoch10_lr5e-4", 
    #            "MNLI_full_global100_eps8_norm1_dp_epoch6_lr5e-4", 
    #            "MNLI_full_global100_eps8_norm10_dp_epoch6_lr5e-4"]
    #NOTE small eps norm ablation
    #models += ["MNLI_full_eps0.1_norm0.1_dp_epoch18_lr5e-4", "MNLI_full_eps0.1_norm1_dp_epoch18_lr5e-4", "MNLI_full_eps0.1_norm10_dp_epoch18_lr5e-4"]
    #NOTE sgld ablation
    # models += ["MNLI_full_sgld_eps8_norm0.1_dp_epoch18_lr5e-4_testlatency", "QNLI_full_sgld_eps8_norm0.1_dp_epoch6_lr1e-3_testlatency", 
    #            "QQP_full_sgld_eps8_norm0.1_dp_epoch18_lr5e-4_testlatency", "SST-2_full_sgld_eps8_norm0.1_dp_epoch8_lr1e-3_testlatency"]
    # NOTE nondp
    # res_folder = "nondp"
    # models += ["MNLI_full_nondp_epoch18_lr5e-5", "QNLI_full_nondp_epoch6_lr1e-4", "QQP_full_nondp_epoch18_lr5e-5", "SST-2_full_nondp_epoch8_lr5e-5", ]
    #models += ["MNLI_full_same0.8281_epoch1_lr5e-5", "QNLI_full_same0.8503_epoch3_lr1e-4", "QQP_full_same0.8685_epoch3_lr5e-5", "SST-2_full_same0.8922_epoch3_lr5e-5"]
    # NOTE reg ablation
    # models += ["MNLI_full_nondp_drop0.2_lr1e-5", "MNLI_full_nondp_drop0.3_lr1e-5", "MNLI_full_nondp_drop0.4lr1e-5", 
    #            "MNLI_full_nondp_early2_lr1e-5", "MNLI_full_nondp_early4_lr1e-5", "MNLI_full_nondp_early6_lr1e-5", 
    #            "MNLI_full_nondp_l2-0.1_lr1e-5", "MNLI_full_nondp_l2-0.01_lr1e-5", "MNLI_full_nondp_l2-1e-3_lr1e-5", "MNLI_full_nondp_l2-1e-4_lr1e-5",
    #            "MNLI_nonal_dp_30w"]
    for model in models:
        print(model)
        default = 'none'
        train = model.split('_')[0].lower()                    
        dp_models = ['sgd'] #, default]  # 'sgd' 'pate', 'an', 'pate', 'obj', 'out'
        #dp_models = [default] # ['sgd'] # default,  # ['sgd']
        test_datasets = test_ls[train]
        #test_datasets = [train]
        for dp_type in dp_models:
            for test in test_datasets: 
                results = []
                print("DP type: {}, Test dataset: {}".format(dp_type, test))
                epsilons = [8.0]
                max_iter = 2 # 100
                num_class = num_class_ls[train]
                recal_type = 'temp'
                recal_type = 'Platt'
                for epsilon in epsilons:
                    for seed in seeds:
                        set_seed(seed=seed)
                        if test.upper() == train.upper():
                            dp_model, netcal_temp, uncalibrated_score, netcal_calibrated_score, dp_calibrated_score = \
                            main(model, max_iter, num_class=num_class, epsilon=epsilon, train_dataset=train, test_dataset=test, dp_type=dp_type, recal_type=recal_type, res_folder=res_folder)
                        else:
                            uncalibrated_score, netcal_calibrated_score, dp_calibrated_score = ood_main(model, dp_model, netcal_temp, train_dataset=train, test_dataset=test, res_folder=res_folder) # in_temp=netcal_temp
                        results.append([uncalibrated_score, netcal_calibrated_score, dp_calibrated_score])
                results = torch.tensor(results)
                mean, std = results.mean(0), results.std(0)
                print("*"*10, f"Final Results for eps {epsilon}", "*"*10)
                print(f"{model}, {dp_type}, {test}")
                print("No calibration, OOD Netcal, OOD DP")
                print("Mean, Std:", mean, std)
                print("*"*35)

'''
# "MNLI_full_eps8_norm0.1_dp_lr1e-4" #, "QNLI_full_eps8_norm0.1_dp_lr1e-3", "SST-2_full_eps8_norm0.1_dp_lr1e-3"]#["MNLI_full_eps3_norm0.1_dp_lr1e-4", "MNLI_full_eps8_norm0.1_dp_lr1e-4", "MNLI_full_eps11_norm0.1_dp_lr1e-4", "MNLI_full_eps25_norm0.1_dp_lr1e-4"]
'''