import numpy

from torch import nn

"""
The full HIV replication inhibition classifier model.
This model is built from an ensemble of PredictionModel objects which were trained on HIV replication
inhibition data.
Note that this model is built from other trained models, so this model does not need to be trained.

Predicts whether a molecule will inhibit HIV replication (0) or not (1).
"""
class HIVClassifier(nn.Module):
    """
    Initializes an object instance of an HIVClassifier model.

    Parameters:
        - models: The set of models to use in an ensembled form for the prediction.
    """
    def __init__(self, models):
        super(HIVClassifier, self).__init__()
        self.models = models
    
    """
    For a given set of molecules, predicts whether the molecules will inhibit HIV replication or not.
    """
    def forward(self, molecules):
        predictions = None

        # Get predictions for each model.
        for model in self.models:
            model.eval()
            y_pred = model(molecules).detach().numpy()

            if predictions == None:
                predictions = numpy.array(y_pred)
            else:
                predictions += y_pred

        # Average predictions and return.
        predictions /= len(self.models)
        return predictions
