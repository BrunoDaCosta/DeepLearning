import torch
from mini_torch import *
from generate_data import *

VERBOSE = 0
torch.set_printoptions(precision=2)

nbiter = 300
nbdata = 300

epsilon = 0.3
eta = 1e-1 / nbdata

train_input, train_label, test_input, test_label = generate_data(nbdata)
mean, std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

model = Sequential(Linear(2,25, epsilon=epsilon), leaky_ReLU(),
                   Linear(25,25, epsilon=epsilon), leaky_ReLU(),
                   Linear(25,25, epsilon=epsilon), leaky_ReLU(),
                   Linear(25, 1, epsilon=epsilon), Sigmoid(),
                   eta=eta)

mod_loss = torch.zeros(nbiter)
xest = torch.empty(nbdata,1)
xest_test = torch.empty(nbdata,1)
errors = torch.empty(nbiter)
errors_test = torch.empty(nbiter)

for i in range(nbiter):
    model.zero_grad()
    # accumulate gradient on all samples then apply it to the weights and offset
    for n in range(nbdata):
        xest[n] = model.forward(train_input[n])
        dloss = dLossMSE(xest[n], train_label[n])
        mod_loss[i] += LossMSE(xest[n], train_label[n])
        model.backward(dloss)
        xest_test[n] = model.forward(test_input[n])
    model.step()

    # Log the loss and train + test error at each iteration for plotting
    mod_loss[i] /= nbdata
    errors[i] = classify(xest, train_label).item() / nbdata * 100
    errors_test[i] = classify(xest_test, test_label).item() / nbdata * 100

    print("###\n{0:.1f}% - MSE :  {1:.4f}".format(i / nbiter * 100, mod_loss[i]))
    if VERBOSE:
        print("        Classification error: {0:.1f}%".format(errors[i]))

# Run the forward pass on test data with the trained model, and print classification error
for n in range(nbdata):
    xest[n] = model.forward(train_input[n])
    xest_test[n] = model.forward(test_input[n])
if VERBOSE:
    print(xest.t())
    print(train_label.t())
print("Train data classification error: {0:.1f}%".format(classify(xest,train_label).item()/nbdata*100))
print("Test data classification error: {0:.1f}%".format(classify(xest_test,test_label).item()/nbdata*100))


# Plot the loss, train+test error rate curves with respect to iterations
print_curves = True
if print_curves:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.subplot(3,1,1)
    plt.plot(mod_loss.numpy(), 'b')
    plt.title("MSE loss")
    plt.subplot(3,1,2)
    plt.plot(errors.numpy(), 'r')
    plt.title("Percentage of classification errors for train")
    plt.subplot(3,1,3)
    plt.plot(errors_test.numpy(), 'g')
    plt.title("Percentage of classification errors for test")
    plt.savefig("latest_data.png") # save the fig as png
    plt.show()



# Create a grid and evaluate all points in it, to reconstruct a separation of space with the trained model
print_separation = False
if print_separation:
    print("Starting boundary reconstruction")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    import numpy as np

    fig, ax = plt.subplots()
    # Make data.
    min = 0
    max = 1
    nbpoints = 100
    X = np.arange(0, nbpoints) / nbpoints * (max-min) + min
    Y = np.arange(0, nbpoints) / nbpoints * (max-min) + min

    Z = np.zeros((nbpoints,nbpoints))
    for i in range(nbpoints):
        for j in range(nbpoints):
            Z[i][j] = model.forward(torch.tensor([float(X[i]), float(Y[j])]).sub_(mean).div_(std))
        #print("Reconstructing: {0:.1f}% done".format(i/nbpoints*100))

    # Plot the raw output, or the classified output
    separate = True
    if separate:
        Z[Z >= 0.5] = 1
        Z[Z < 0.5] = 0

    # Plot the result
    surf = ax.imshow(Z, extent = [min, max, min, max])

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
