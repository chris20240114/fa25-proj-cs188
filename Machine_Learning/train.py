from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        "*** YOUR CODE HERE ***"
        while True:
            any_mistake = False
            for batch in dataloader:
                x = batch['x'].view(-1)
                y = int(batch['label'].item())
                pred = model.get_prediction(x)
                if pred != y:
                    any_mistake = True
                    model.w.data += (y * x).view(model.w.data.shape)
            if not any_mistake:
                break


def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    "*** YOUR CODE HERE ***"
    model.train()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    max_epochs = 2000
    target_loss = 0.02

    for epoch in range(max_epochs):
        running_loss = 0.0
        count = 0
        for batch in dataloader:
            x = batch['x']
            y = batch['label']
            optimizer.zero_grad()
            y_pred = model(x)
            loss = regression_loss(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.shape[0]
            count += x.shape[0]

        avg_loss = running_loss / max(1, count)
        if avg_loss <= target_loss:
            break


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    """ YOUR CODE HERE """
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    max_epochs = 20
    val_threshold = 0.975

    for epoch in range(max_epochs):
        for batch in dataloader:
            x = batch['x']
            y = batch['label']
            
            optimizer.zero_grad()
            preds = model(x)
            loss = digitclassifier_loss(preds, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with no_grad():
            val_acc = dataset.get_validation_accuracy()
        model.train()

        print(f"Epoch {epoch + 1}: Validation Accuracy = {val_acc:.4f}")

        if val_acc >= val_threshold:
            print(f"Terminate: Validation Accuracy reached {val_acc:.4f}.")
            break

def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    "*** YOUR CODE HERE ***"
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    max_epochs = 20
    val_threshold = 0.81

    for epoch in range(max_epochs):
        for batch in dataloader:
            x = batch['x']
            y = batch['label']
            x_seq = movedim(x, 0, 1)
            
            optimizer.zero_grad()
            preds = model(x_seq)
            loss = languageid_loss(preds, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with no_grad():
            val_acc = dataset.get_validation_accuracy()
        model.train()
        print(f"Epoch {epoch+1}: Validation Accuracy: {val_acc}")

        if val_acc >= val_threshold:
            break



def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """
    model.train()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    max_epochs = 20
    val_threshold = 0.80

    for epoch in range(max_epochs):
        for batch in dataloader:
            x = batch['x']
            y = batch['label']
            
            optimizer.zero_grad()
            preds = model(x)
            loss = digitconvolution_Loss(preds, y)
            loss.backward()
            optimizer.step()
        model.eval()
        with no_grad():
            val_acc = dataset.get_validation_accuracy()
        model.train()

        print(f"Epoch {epoch + 1}: Validation Accuracy = {val_acc:.4f}")

        if val_acc >= val_threshold:
            print(f"Terminate: Validation Accuracy reached {val_acc:.4f}.")
            break
