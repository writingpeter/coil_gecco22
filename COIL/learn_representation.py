import importlib
import argparse
import numpy as np
import os, time
import pickle
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing

# directories
DATA_DIRECTORY = 'data/'
VAE_DIRECTORY = 'vae/'

# https://github.com/pytorch/examples/blob/master/vae/main.py
class VecVAE(nn.Module):
    def __init__(self, input_space, latent_space, dataset=None):
        super(VecVAE, self).__init__()
        if dataset == None:
          self.dataset = VecDataSet # Dataset of 1d vectors
        else:
          self.dataset = dataset

        self.input_space = input_space
        self.latent_space = latent_space
        self.hidden_size = latent_space * 2


        self.fc1  = nn.Linear(self.input_space, self.hidden_size)
        self.fc21 = nn.Linear(self.hidden_size, latent_space)
        self.fc22 = nn.Linear(self.hidden_size, latent_space)
        self.fc3  = nn.Linear(latent_space    , self.hidden_size)
        self.fc4  = nn.Linear(self.hidden_size, self.input_space)

        self.optimizer =  optim.Adam(self.parameters(), lr=1e-3)

        self.loss_function = self.loss # to easily change between loss functions


    def save(self, path):
        torch.save(self.state_dict(), path)

    # - Moving data through ------------- - #
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def express(self,z):
        # Use VAE as a generator, given numpy latent vec return full numpy phenotype
        latent = torch.from_numpy(z).float()
        pheno = self.decode(latent)
        return pheno.detach().numpy()

    def reconstruct(self, x):
        # Run through encoder and decoder
        l = self.encode(torch.from_numpy(x).float())[0]
        r = np.reshape(self.decode(l).detach().numpy(), x.shape)
        recon = r
        return recon


    # - Loss Functions ------------- - #
    def loss(self, recon_x, x, mu, logvar, input_space, kl_weight=1.0):
        # print(x)
        # print(recon_x)
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
        # return BCE + (0.1 * KLD)
        # return BCE + (0.5 * KLD)
        return BCE + (1.0 * KLD)


    # - Training ------------------ - #
    def fit(self, dataloader, n_epoch, viewMod=100, returnLoss=False):
        loss = np.full(n_epoch+1, np.nan)
        self.train()
        for e in range(1, n_epoch+1):
            mean,std = self.epoch(e, dataloader, e/(n_epoch+1))
            loss[e] = mean
            if ((e) % viewMod) == 0:
                if debug:
                    print('Loss at Epoch ', e, ':\t', mean)
        if returnLoss:
            return loss

    def epoch(self, epoch_id, train_loader, percDone):
        self.train()
        train_loss = []
        kl_weight = np.min([percDone*4.0,1.0])
        for batch_idx, (data, fit) in enumerate(train_loader):
            data = data.to(device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.forward(data)
            loss = self.loss_function(recon_batch, data, mu, logvar, self.input_space, kl_weight)
            loss.backward()
            r = np.linalg.norm(recon_batch.detach().numpy() - data.detach().numpy(), axis=0)
            train_loss += [r]
            self.optimizer.step()
        per_input_loss = np.vstack(train_loss)
        return np.mean(per_input_loss), np.std(per_input_loss)   

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar     

class VecDataSet(Dataset):
    def __init__(self, pop):
        self.data = torch.tensor(pop).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx,:], 0


def train_vae(genomes, vae, n_epochs, view_mod=100):
    """ Simple script to train the VAE:
    genomes : an [n_individuals x n_parameters] matrix
    vae     : a vae created with the VecVAE class in this file
    n_epochs: how long to train
    """
    dataset = vae.dataset(genomes)
    training_batch = min(genomes.shape[0], 64)
    dataloader = DataLoader(dataset, batch_size=training_batch, shuffle=True)        
    with torch.enable_grad():
        loss = vae.fit(dataloader, n_epochs, viewMod=view_mod, returnLoss=True)  
    return vae, loss


def load_data(directory, equation_name, debug):

    data_list = []

    # load data for each constraint
    for i in range(len(eq.CONSTRAINTS)):
        current_file = directory + equation_name + '_v' + str(NUM_VARIABLES) + '_constraint' + str(i) + '.pkl'
        print('Load data: ', current_file)
        generate_data_seed, data = pickle.load(open(current_file, 'rb'))
        data_list += data

    raw_data = np.array(data_list)
    if debug:
          print(raw_data[:10])
          # print(raw_data.mean(axis=0))
          # print(raw_data.std(axis=0))
          print('raw data min max', np.min(raw_data), np.max(raw_data), )

    scaler = preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0)).fit(raw_data)
    genomes = scaler.transform(raw_data)
    if debug:
        print('scaled data min max', np.min(genomes), np.max(genomes))
        print(genomes[:10])
    return genomes, scaler


#------------------------------------------------------------------------------#
if __name__ == "__main__":
    #----------
    # Get command line arguments
    #----------
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--equation', help='equation name should correspond to the equation file, e.g., enter eq01 to import the equation file eq01.py')
    parser.add_argument('-v', '--num_variables', help='number of variables')
    parser.add_argument('-b', '--best_out_of', help='default is 10')
    parser.add_argument('-s', '--seed', help='enter -1 for a random seed, or enter the seed number. if the argument is not provided, it defaults to -1.')
    parser.add_argument('-d', '--debug', action='store_true', help='if argument is used, generate debug info.')
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    
    args = parser.parse_args()

    device = 'cpu'
    # args.cuda = not args.no_cuda and torch.cuda.is_available()

    #----------
    # Import equation module
    #----------
    if args.equation:
        equation_name = args.equation
        eq = importlib.__import__(equation_name) # import equation module
    else:
        exit("Error. Please specify equation name in the command line. Use --help for more information.")

    #----------
    # Get number of variables
    #----------
    if args.num_variables:
        NUM_VARIABLES = int(args.num_variables)
        if NUM_VARIABLES < eq.MIN_VARIABLES:
            exit("Error. Minimum number of variables for this function is %d. Use --help for more information." % eq.MIN_VARIABLES)
    else:
        NUM_VARIABLES = eq.MIN_VARIABLES
    NUM_LATENT = NUM_VARIABLES

    #----------
    # Set seed
    #----------
    if not args.seed or int(args.seed) < 0: # if args.seed is not provided or is negative
        seed = int(time.time()) # use current time as random seed
    else:
        seed = int(args.seed)
    print('Seed', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    #----------
    # Set flags
    #----------
    debug = args.debug

    if args.best_out_of:
        BEST_OUT_OF = int(args.best_out_of)
    else:
        BEST_OUT_OF = 10 # save 1 vae out of 10 runs with the lowest loss

    genomes, scaler = load_data(DATA_DIRECTORY, equation_name, debug)

    n_dim = genomes.shape[1]

    list_of_vaes = []
    list_of_loss = []

    for i in range(BEST_OUT_OF):
        # Create VAE
        if debug: 
            print('Run', i)
        vae = VecVAE(n_dim, NUM_LATENT)

        # Train VAE
        vae, loss = train_vae(genomes, vae, eq.NUM_EPOCHS, view_mod=2)
        list_of_vaes.append(vae)
        final_loss_value = loss[-1]
        if debug:
            print('Loss', final_loss_value)
        list_of_loss.append(final_loss_value)

    min_loss_value = min(list_of_loss)
    min_loss_index = list_of_loss.index(min_loss_value)
    print('Best min loss', min_loss_value)
    vae = list_of_vaes[min_loss_index] # get the best vae with the lowest loss

    # Save VAE and Scaler
    if not os.path.exists(VAE_DIRECTORY):
        os.makedirs(VAE_DIRECTORY)

    vae_file = VAE_DIRECTORY + equation_name + '_v' + str(NUM_VARIABLES) + '_vae.pt'
    vae.save(vae_file)
    print("VAE saved to", vae_file)

    scaler_file = VAE_DIRECTORY + equation_name + '_v' + str(NUM_VARIABLES) + '_vae_scaler.pkl'
    pickle.dump([seed, scaler], open(scaler_file, 'wb'))
    print("Seed and scaler saved to", scaler_file)


    if debug:
        # Use VAE as a representation
        n_ind = 1
        random_genome = np.random.randn(n_ind, NUM_LATENT)
        sampled_point = vae.express(random_genome)
        print('random genome', random_genome)
        print('sample point', sampled_point)
        transformed = scaler.transform(sampled_point)
        print('transformed point', transformed)
        print('scaler transform back', scaler.inverse_transform(transformed))

        not_so_random_genome = np.zeros([n_ind, NUM_LATENT])
        evolved_point = vae.express(not_so_random_genome)

        existing_solution = np.zeros([n_ind, n_dim])
        reconstructed_point = vae.reconstruct(existing_solution)
        print('test reconstruct')
        print(existing_solution)
        print(reconstructed_point)
        print('scaler transform back', scaler.inverse_transform(reconstructed_point))
