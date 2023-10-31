import torch
from torch.utils.data import DataLoader, Dataset
from generator import DataGenerator
from joblib import Parallel, delayed
from multiprocessing.dummy import Pool


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, num_items):
        'Initialization'
        self.num_items = num_items
        # self.pool = ThreadPool(4)

  def __len__(self):
        'Denotes the total number of samples'
        return self.num_items

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        
        # Load data and get label

# Define some dummy data
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 1}

max_epochs = 100
device = 1

# Create a DataLoader
# Generators
training_set = Dataset(5)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(3)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        [...]

    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            [...]

# Iterate over the DataLoader to get batches of data
# for batch in dataloader:
#     print(batch.shape)

####################################################################################################

# import torch
# from my_classes import Dataset


# class Dataset(torch.utils.data.Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, list_IDs, labels):
#         'Initialization'
#         self.labels = labels
#         self.list_IDs = list_IDs

#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)

#   def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         ID = self.list_IDs[index]

#         # Load data and get label
#         X = torch.load('data/' + ID + '.pt')
#         y = self.labels[ID]

#         return X, y

# params = {'batch_size': 64,
#           'shuffle': True,
#           'num_workers': 6}

# # CUDA for PyTorch
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True

# # Parameters

# max_epochs = 100

# # Datasets
# partition = # IDs
# labels = # Labels

# # Generators
# training_set = Dataset(partition['train'], labels)
# training_generator = torch.utils.data.DataLoader(training_set, **params)

# validation_set = Dataset(partition['validation'], labels)
# validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# # Loop over epochs
# for epoch in range(max_epochs):
#     # Training
#     for local_batch, local_labels in training_generator:
#         # Transfer to GPU
#         local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#         # Model computations
#         [...]

#     # Validation
#     with torch.set_grad_enabled(False):
#         for local_batch, local_labels in validation_generator:
#             # Transfer to GPU
#             local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#             # Model computations
#             [...]