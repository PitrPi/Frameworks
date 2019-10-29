import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


def test_linear(model, test_data):
    with torch.no_grad():  # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
        else:
            predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    return predicted


def confusion(target, prediction):
    cm = confusion_matrix(target, prediction)
    return cm

