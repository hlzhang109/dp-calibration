from classification.utils import *
import pandas as pd
from tqdm import tqdm
import csv

label_list = ['contradiction', 'neutral', 'entailment']
label_map = {label: i for i, label in enumerate(label_list)}
map_label = {i: label for i, label in enumerate(label_list)}
corruption_type = 'unif'
num_classes = 3

def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def corrupt_uniform(x, num_classes, C):
    size = len(x)
    for i in range(size):
        x[i] = np.random.choice(num_classes, p=C[x[i]])
    return x 

# NOTE generate data before hand for the label noise experiments.
corr_list = [0.2, 0.4, 0.6, 0.8, 1.0]
for corruption_prob in tqdm(corr_list):
    C = uniform_mix_C(corruption_prob, 3)
    corruptor = Corruptor(num_classes=3, corruption_type=corruption_type, corruption_prob=corruption_prob)
    print(f" *** corruption init {corruption_type}: {corruption_prob}")
    quotechar=None
    input_file = 'classification/data/original/MNLI/train.tsv'
    with open(input_file, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter="\t", quotechar=quotechar))
    for i, d in tqdm(enumerate(data)):
        if i == 0:
            continue
        label1, gold_label = label_map[d[-2]], label_map[d[-1]] #d['label1'], d['gold_label']
        corrupted_label = corrupt_uniform([gold_label], num_classes=num_classes, C=C)[0]
        print(corrupted_label, gold_label)
        gold_label = corrupted_label
        data[i][-1] = data[i][-2] = map_label[gold_label]
    
    output_file = f'classification/data/original/MNLI{corruption_prob}/train.tsv'
    output = pd.DataFrame(data[1:], columns=data[0])
    output.to_csv(output_file, sep='\t', index=False, encoding="utf-8-sig")