import numpy

from torch import nn
from torch.optim import Adam

from .directed_mpnn import DMPNNEncoder

from chemprop.nn_utils import NoamLR

from sklearn.metrics import f1_score, roc_auc_score

"""
The full classifier model which uses a D-MPNN model to create an embedding for the
input molecule, as well as a set of provided molecular features, to predict the classification
of that molecule.
"""
class PredictionModel(nn.Module):
    """
    Initializes an object instance of a PredictionModel.

    Parameters:
        - args : An instance of the ModelArgs class which contains various parameters for the model.
        - atom_dim : Atom feature vector dimension.
        - bond_dim : Bond feature vector dimension.
        - features_dim : The dimension of the additional feature vector.
        - torch_device : The PyTorch device to use with the model.
    """
    def __init__(self, args, atom_dim, bond_dim, features_dim, torch_device):
        super(PredictionModel, self).__init__()

        # Encoder model, which will encode the input molecule.
        self.encoder = DMPNNEncoder(args, atom_dim, bond_dim).to(torch_device)

        # Activation function for the feed-forward neural networks which will compute prediction.
        self.sigmoid = nn.Sigmoid()

        # Dropout layer to use for feed-forward neural networks.
        dropout = nn.Dropout(args.ffn_dropout_prob)

        # The layers of feed-forward neural networks which will compute the property 
        # prediction from the embedding.
        if (args.num_ffn_layers == 1):
            ffn = [dropout, nn.Linear(args.hidden_size + features_dim, 1)]
        else:
            # First layer.
            ffn = [dropout, nn.Linear(args.hidden_size + features_dim, args.ffn_hidden_size)]

            # Middle layers.
            for _ in range(args.num_ffn_layers - 2):
                ffn.extend([self.sigmoid, dropout, nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)])
            
            # Final layer.
            ffn.extend([self.sigmoid, dropout, nn.Linear(args.ffn_hidden_size, 1)])
        
        # Create final FFN model.
        self.ffn = nn.Sequential(*ffn).to(torch_device)

        # Initialize the weights for the model.
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    """
    Predicts whether a given molecule has a specific property or not.
    """
    def forward(self, batch_molecules):
        # Compute prediction.
        output = self.ffn(self.encoder(batch_molecules))

        # Only apply sigmoid to output when not training, as we will use BCEWithLogitsLoss
        # for training which will apply a sigmoid to the output.
        if not self.training:
            output = self.sigmoid(output)
        
        return output


"""
Trains the prediction model for a given data loader.
"""
def train_prediction_model(model, data_loader, criterion, num_epochs):
    # Get optimizer and learning rate scheduler.
    optimizer = Adam([{"params": model.parameters(), "lr": 1e-4, "weight_decay": 0}])
    scheduler = NoamLR(optimizer=optimizer, warmup_epochs=[2.0], total_epochs=[num_epochs],
        steps_per_epoch=len(data_loader), init_lr=[1e-4], max_lr=[1e-3], final_lr=[1e-4])
    
    # Train.
    model.train()

    loss_sum = 0
    for data in data_loader:
        optimizer.zero_grad()
        y_hat = model(data)
        loss = criterion(y_hat, data.y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    print("Current prediction model loss: " + str(loss_sum / len(data_loader)))


"""
Tests the prediction model on a given data loader.
"""
def test_prediction_model(model, data_loader):
    model.eval()

    # Test.
    y_pred = []
    y_pred_labels = []
    y_true = []
    for data in data_loader:
        y_hat = model(data).numpy()
        y_pred += y_hat.tolist()
        y_pred_labels += numpy.round(y_hat).tolist()
        y_true += data.y.numpy().tolist()

    # Compute metrics.
    f1 = f1_score(y_true, y_pred_labels)
    roc_auc = roc_auc_score(y_true, y_pred)
    return f1, roc_auc
