import torch
from mini_torch import *
from generate_data import *

VERBOSE = 0

nbiter = 100
nbdata = 300

epsilon = 0.3
eta = 1e-1 / nbdata

train_input, train_label, test_input, test_label = generate_data(nbdata)
mean, std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

model = sequential(linear(2,25, epsilon=epsilon), leaky_relu(), linear(25,25, epsilon=epsilon), leaky_relu(), linear(25,25, epsilon=epsilon), leaky_relu(),linear(25, 1, epsilon=epsilon), sigmoid(), eta = eta)

mod_loss = torch.zeros(nbiter)
xest = torch.empty(nbdata,1)
xest_test = torch.empty(nbdata,1)
errors = torch.empty(nbiter)
errors_test = torch.empty(nbiter)

for i in range(nbiter):
    #eta = eta_start * (1 - (i/nbiter))
    print(model.param()[-1])
    model.zero_grad()
    print(model.param()[-1])
    print("########")
    for n in range(nbdata):
        xest[n] = model.forward_pass(train_input[n])
        #print(train_input[n], yest)
        dloss = MSEdloss(xest[n], train_label[n])
        #print(xest[n])
        #print(train_label[n])
        mod_loss[i] += MSEloss(xest[n], train_label[n])
        model.backward_pass(dloss)

        xest_test[n] = model.forward_pass(test_input[n])
    model.step()

    mod_loss[i] /= nbdata
    errors[i] = classify(xest, train_label).item() / nbdata * 100
    errors_test[i] = classify(xest_test, test_label).item() / nbdata * 100

    print("###\n{0:.1f}% - MSE :  {1}".format(i / nbiter * 100, mod_loss[i]))
    if VERBOSE:
        print("        Classification error: {0:.1f}%".format(errors[i]))

torch.set_printoptions(precision=2)

for n in range(nbdata):
    xest[n] = model.forward_pass(train_input[n])
if VERBOSE:
    print(xest.t())
    print(train_label.t())
print("Classification error:\n {0:.1f}%\n".format(classify(xest,train_label).item()/nbdata*100))
#print(model.param())

print_curves = 0
if print_curves:
    import matplotlib.pyplot as plt
    def close_event():
        plt.close()
    fig = plt.figure()
    timer = fig.canvas.new_timer(interval=10000)
    #timer.add_callback(close_event)
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
    #timer.start()
    plt.show()


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

print_separation = 1
if print_separation:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Make data.
    minmax=2
    nbpoints = 100
    X = np.arange(-minmax, minmax, 2*minmax/nbpoints)
    Y = np.arange(-minmax, minmax, 2*minmax/nbpoints)
    X, Y = np.meshgrid(X, Y)

    Z = np.zeros((nbpoints,nbpoints))
    for i in range(nbpoints):
        for j in range(nbpoints):
            print(str(100 * (i * nbpoints + j) / (nbpoints * nbpoints)) + "% done")
            #print(model.forward_pass(torch.tensor([i/(nbpoints/(2*minmax))-minmax, j/(nbpoints/(2*minmax))-minmax])))
            Z[i][j] = model.forward_pass(torch.tensor([i/(nbpoints/(2*minmax))-minmax, j/(nbpoints/(2*minmax))-minmax]))

    Z[Z >= 0.5] = 1
    Z[Z < 0.5] = 0
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.5, 1.5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
