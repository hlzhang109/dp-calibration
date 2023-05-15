import numpy as np

#from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import pickle
import torch
import torch.nn as nn


def label_mappings():
    SST2_label_map = {'negative':0, 'positive':1}
    COLA_label_map = {'unacceptable':0, 'acceptable':1}

    WNLI_label_map = {'entailment':1, 'not_entailment': 0} # 1 (entailment), 0 (not_entailment)
    QQP_label_map = {'duplicate':1, 'not_duplicate': 0} # 1 (entailment), 0 (not_entailment)
    MRPC_label_map = {'equivalent':1, 'not_equivalent': 0} # 1 (entailment), 0 (not_entailment)

    RTE_label_map =  {'entailment':0, 'not_entailment': 1} # 1 -- equivalent, 0 -- not equivalent.
    QNLI_label_map = {'entailment':0, 'not_entailment': 1} # 1 (not_entailment), 0 (entailment), label_list =  ["entailment", "not_entailment"]
    HANS_label_map = {'entailment':0, 'not_entailment': 1} # 1 (not_entailment), 0 (entailment), label_list =  ["entailment", "not_entailment"]
    SCITAIL_label_map = {'entailment':0, 'not_entailment': 1} # 1 (not_entailment), 0 (entailment), label_list =  ["entailment", "not_entailment"]

    MNLI_label_map = {'contradiction':0, 'neutral':1, 'entailment': 2} 
    return
    
class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels): # logits  # softmaxes
        #softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=labels.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C

def flip_labels_C_two(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2
    return C

class Corruptor:
    def __init__(self, num_classes, corruption_type, corruption_prob=0):
        self.corruption_type = corruption_type
        #self.matrix = matrix
        self.corruption_prob = corruption_prob
        self.num_classes = num_classes

        if corruption_type == 'unif':
            C = uniform_mix_C(self.corruption_prob, num_classes)
            print(C)
            self.C = C
        elif corruption_type == 'flip':
            C = flip_labels_C(self.corruption_prob, num_classes)
            print(C)
            self.C = C
        elif corruption_type == 'flip2':
            C = flip_labels_C_two(self.corruption_prob, num_classes)
            print(C)
            self.C = C

    def corrupt(self, x):
        if self.corruption_type == 'unif':
            return self.corrupt_uniform(x)
        elif self.corruption_type == 'flip':
            return self.corrupt_flip(x)
        elif self.corruption_type == 'flip_two':
            return self.corrupt_flip_two(x)
        else:
            raise ValueError('Unknown corruption type')

    def corrupt_uniform(self, x):
        size = len(x)
        for i in range(size):
            x[i] = np.random.choice(self.num_classes, p=self.C[x[i]])
        return x 
    
    # TODO implement
    def corrupt_flip(self, x):
        return x * self.C
    
    def corrup_flip_two(self, x):
        return x * self.C

        if corruption_type == 'hierarchical':
            self.train_coarse_labels = list(np.array(train_coarse_labels)[idx_to_train])


        elif corruption_type == 'hierarchical':
            assert num_classes == 100, 'You must use CIFAR-100 with the hierarchical corruption.'
            coarse_fine = []
            for i in range(20):
                coarse_fine.append(set())
            for i in range(len(self.train_labels)):
                coarse_fine[self.train_coarse_labels[i]].add(self.train_labels[i])
            for i in range(20):
                coarse_fine[i] = list(coarse_fine[i])

            C = np.eye(num_classes) * (1 - corruption_prob)

            for i in range(20):
                tmp = np.copy(coarse_fine[i])
                for j in range(len(tmp)):
                    tmp2 = np.delete(np.copy(tmp), j)
                    C[tmp[j], tmp2] += corruption_prob * 1/len(tmp2)
            self.C = C
            print(C)
        elif corruption_type == 'clabels':
            net = wrn.WideResNet(40, num_classes, 2, dropRate=0.3).cuda()
            model_name = './cifar{}_labeler'.format(num_classes)
            net.load_state_dict(torch.load(model_name))
            net.eval()
        else:
            assert False, "Invalid corruption type '{}' given. Must be in {'unif', 'flip', 'hierarchical'}".format(corruption_type)
        
        if corruption_type == 'clabels':
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]

            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])

            # obtain sampling probabilities
            sampling_probs = []
            print('Starting labeling')

            for i in range((len(self.train_labels) // 64) + 1):
                current = self.train_data[i*64:(i+1)*64]
                current = [Image.fromarray(current[i]) for i in range(len(current))]
                current = torch.cat([test_transform(current[i]).unsqueeze(0) for i in range(len(current))], dim=0)

                data = V(current).cuda()
                logits = net(data)
                smax = F.softmax(logits / 5)  # temperature of 1
                sampling_probs.append(smax.data.cpu().numpy())

            sampling_probs = np.concatenate(sampling_probs, 0)
            print('Finished labeling 1')

            new_labeling_correct = 0
            argmax_labeling_correct = 0
            for i in range(len(self.train_labels)):
                old_label = self.train_labels[i]
                new_label = np.random.choice(num_classes, p=sampling_probs[i])
                self.train_labels[i] = new_label
                if old_label == new_label:
                    new_labeling_correct += 1
                if old_label == np.argmax(sampling_probs[i]):
                    argmax_labeling_correct += 1
            print('Finished labeling 2')
            print('New labeling accuracy:', new_labeling_correct / len(self.train_labels))
            print('Argmax labeling accuracy:', argmax_labeling_correct / len(self.train_labels))
        else:    
            for i in range(len(self.train_labels)):
                self.train_labels[i] = np.random.choice(num_classes, p=C[self.train_labels[i]])
            self.corruption_matrix = C