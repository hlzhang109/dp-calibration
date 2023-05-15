import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
from tqdm import tqdm
import torch
from opacus.privacy_engine import PrivacyEngine
from torch.utils.data import Dataset, DataLoader
from os.path import join
import random
from torchmetrics import HingeLoss
import argparse
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

def set_seed(seed=0):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--grad_norm", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=int(1e4))
    parser.add_argument("--weight_type", type=str, default="No IW")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--res_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="data") # syn_results/
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--b", type=float, default=1.5)
    parser.add_argument("--batch_size", type=int, default=4000)
    parser.add_argument("--class_two_num", type=int, default=5000)
    parser.add_argument("--class_one_num", type=int, default=5000)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--dp", type=bool, default=False)
    parser.add_argument("--load", type=bool, default=False)
    parser.add_argument("--log_loss", type=bool, default=False)
    parser.add_argument("--mse_loss", type=bool, default=False)
    parser.add_argument("--hinge_loss", type=bool, default=False)

    args = parser.parse_args()
    args.res_path = "DP" if args.dp else "NonDP" 
    args.res_path = join("results", args.res_path)
    print(args)
    print(args.res_path)
    return args


def truncate_normal(input: torch.Tensor, radius):
    in_norm = input.norm(dim=1, keepdim=True)
    in_norm[in_norm > radius] = radius
    return input * in_norm / input.norm(dim=1, keepdim=True)

def gen_data(n_samples):
    b = args.b
    red = torch.randn(args.class_one_num, args.dim) + torch.tensor([0, b]) 
    blue = torch.randn(args.class_two_num, args.dim) + torch.tensor([b, 0])  
    return red, blue

def model_plot(model,X,y,title,args):
    sns.set_style("darkgrid")
    parm = {}
    b = []
    for name, param in model.named_parameters():
        parm[name]=param.detach().numpy()  
    
    w = parm['_module.linear.weight'][0] if args.dp else parm['linear.weight'][0]
    b = parm['_module.linear.bias'][0] if args.dp else parm['linear.bias'][0] 
    plt.scatter(X[:, 0], X[:, 1], alpha=.3, c=y,cmap='jet')
    u = np.linspace(X[:, 0].min(), X[:, 0].max(), 2)
    plt.plot(u, (0.5-b-w[0]*u)/w[1])
    plt.xlim(X[:, 0].min()-0.5, X[:, 0].max()+0.5)
    plt.ylim(X[:, 1].min()-0.5, X[:, 1].max()+0.5)
    savepath = f'{folder}/'+title+'.png'
    print(savepath)
    plt.savefig(savepath)
    plt.show()
    plt.clf()

    logits = model(X).squeeze().detach().numpy()
    y = y.detach().numpy()
    np.save(f'{folder}/'+title+'_logits.npy',logits)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

if __name__ == '__main__':
    set_seed()
    args = arg_parse()
    args.data_path = join(args.data_path, str(args.b))
    n_samples = args.n_samples


    red, blue = gen_data(n_samples)
    folder = f'DP/{args.grad_norm}' if args.dp else 'NonDP'
    folder = join("results", folder)
    if not args.log_loss:
        if args.hinge_loss:
            folder = join('Hinge', folder)
        else:
            folder = join('MSE', folder)
    else:
        folder = join('Log', folder)

    print(folder)

    red_labels = np.zeros(len(red))
    blue_labels = np.ones(len(blue))

    labels = np.append(red_labels,blue_labels)
    inputs = np.concatenate((red,blue),axis=0)

    X_train, X_test, y_train,  y_test = train_test_split(
        inputs, labels, test_size=0.33, random_state=42)

    if args.load:
        X_train, X_test, y_train,  y_test = np.load(f'{args.data_path}/TrainData_X.npy'), np.load(f'{args.data_path}/TestData_X.npy'), \
                                            np.load(f'{args.data_path}/TrainData_y.npy'), np.load(f'{args.data_path}/TestData_y.npy')
        print("Data loaded {}".format(args.data_path))
    else:
        np.save(f'{args.data_path}/TrainData_X.npy', X_train)
        np.save(f'{args.data_path}/TestData_X.npy', X_test)
        np.save(f'{args.data_path}/TrainData_y.npy', y_train)
        np.save(f'{args.data_path}/TestData_y.npy', y_test)
        print("Data saved {}".format(args.data_path))

    epochs = args.num_epochs 
    input_dim = 2 
    output_dim = 1 
    learning_rate = 0.01

    model = LogisticRegression(input_dim,output_dim)

    criterion = torch.nn.BCELoss() if args.log_loss else torch.nn.MSELoss()
    if args.hinge_loss:
        criterion = HingeLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(y_train),torch.Tensor(y_test)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)

    if args.dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            epochs=epochs,
            target_epsilon=3,
            target_delta=len(X_train) ** -1.1,
            max_grad_norm=args.grad_norm,
        )

    losses = []
    losses_test = []
    Iterations = []
    iter = 0
    for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):
        x = X_train
        labels = y_train
        for xs, ys in data_loader:
            xs, ys = xs, ys 
            optimizer.zero_grad() 
            outputs = model(xs) 
            loss = criterion(torch.squeeze(outputs), ys) 
            
            loss.backward() 
            optimizer.step() 

        iter+=1
        if iter%10000==0:
            # calculate Accuracy
            with torch.no_grad():
                correct_test = 0
                total_test = 0
                outputs_test = torch.squeeze(model(X_test))
                loss_test = criterion(outputs_test, y_test)
                
                predicted_test = outputs_test.round().detach().numpy()
                total_test += y_test.size(0)
                correct_test += np.sum(predicted_test == y_test.detach().numpy())
                accuracy_test = 100 * correct_test/total_test
                losses_test.append(loss_test.item())
                
                # Calculating the loss and accuracy for the train dataset
                total = 0
                correct = 0
                total += y_train.size(0)
                correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_train.detach().numpy())
                accuracy = 100 * correct/total
                losses.append(loss.item())
                Iterations.append(iter)
                
                print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
                print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")

    # # Train Data
    # model_plot(model,X_train,y_train,'TrainData', args)

    # # Test Dataset Results
    # model_plot(model,X_test,y_test,'TestData', args)