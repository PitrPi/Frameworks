import torch
from torch.autograd import Variable


class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


def train_linear(train_data):
    input_dim = train_data.shape[1]-1  # takes variable 'x'
    output_dim = 1  # takes variable 'y'
    learning_rate = 0.01
    epochs = 100

    model = LinearRegression(input_dim, output_dim)
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(train_data.iloc[:, :-1]).cuda())
            labels = Variable(torch.from_numpy(train_data.iloc[:, -1]).cuda())
        else:
            inputs = Variable(torch.from_numpy(train_data.iloc[:, :-1]))
            labels = Variable(torch.from_numpy(train_data.iloc[:, -1]))

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        print(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))
    return model