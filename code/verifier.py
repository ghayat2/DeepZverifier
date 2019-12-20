import argparse
import torch
import torch.optim as optim
from networks import FullyConnected, Conv, Normalization
import torch.nn.functional as F
import time
import warnings

DEVICE = 'cpu'
INPUT_SIZE = 28
DEBUG = False


def analyze(net, inputs, eps, true_label, model):
    """
This method verifies that the image is still correctly classified when perturbed by noise
    :param net: The network
    :param inputs: the image to verifiy
    :param eps: the magnitude of noise with which the image is perturbed
    :param true_label: the label of the image
    :return: 1 if the image is verified under eps-noise , 0 else
    """
    # set autograd for model parameters to false
    for param in net.parameters():
        param.requires_grad = False

    # Ignoring flatten layers
    layers = [layer for layer in net.layers if not isinstance(layer, torch.nn.Flatten)]
    # Counting the number of ReLU layers
    total_num_relu = sum([1 for layer in net.layers if isinstance(layer, torch.nn.ReLU)])
    # Check if layers must be frozen (Convolutional network must have all but last layer frozen while Dense network
    # should have none frozen)

    # parameters we get for each case
    learning_rate, threshold = get_training_parameters(model, layers, total_num_relu)

    freeze = -threshold is not total_num_relu

    # Building the zonotope
    inputs = inputs.reshape(-1)
    initial_zonotope = build_zonotope(inputs, eps)

    # In the case of convolutional networks, the zonotope before the last ReLU layer will be saved to avoid
    # re-computation at every run
    saved_zonotope = initial_zonotope

    slope_set = []
    number_runs = 0
    start = time.time()
    parameters = list(net.parameters())

    prev_loss = 0

    max_time = 60 * 2

    # Loop until the zonotope is verified or a time out exception occurs

    while time.time() - start < max_time:
        num_relu = 0
        img_dim = INPUT_SIZE

        for i in range(len(layers)):
            layer = layers[i]

            if isinstance(layer, Normalization):
                mean = layer.mean.reshape(-1)[0]
                sigma = layer.sigma.reshape(-1)[0]
                sdt = torch.eye(inputs.size(0)) * (1 / sigma)
                mean = torch.ones(size=inputs.size()) * (-mean / sigma)

                zonotope = affine_dense(initial_zonotope, sdt, mean)

            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
                weight_matrix = parameters[i - 1].type(torch.float32)
                bias = parameters[i].type(torch.float32)

                if isinstance(layer, torch.nn.Linear):
                    zonotope = affine_dense(zonotope, weight_matrix, bias)
                else:
                    zonotope = affine_conv(zonotope, layer, weight_matrix, bias, img_dim)
                    img_dim = img_dim // layer.stride[0]

                # Saving the zonotope before last ReLU layer to avoid re-computation
                if number_runs is 0 and freeze and layer not in layers[threshold:]:
                    saved_zonotope = zonotope.detach()

            if isinstance(layer, torch.nn.ReLU):

                if number_runs is not 0 and freeze:
                    zonotope = saved_zonotope

                (l, u) = compute_upper_lower_bounds(zonotope)
                if number_runs is 0:
                    u, l = u.detach(), l.detach()
                    slopes = freeze_all_but_last(threshold, num_relu, total_num_relu, l, u)
                    slope_set.append(slopes)

                zonotope = relu(zonotope, l, u, slopes=slope_set[num_relu])
                num_relu = num_relu + 1

        # Verifying stopping condition
        if verify(zonotope, true_label):
            return 1

        # Define the optimizer and slopes to train
        optimizer = optim.Adam(slope_set, lr=learning_rate)

        (l, u) = compute_upper_lower_bounds(zonotope)
        # Calculate loss
        """ Exponential loss """
        diff = u - l[true_label]
        diff[true_label] = 0
        poly = diff + 1

        # loss = (torch.sum(torch.exp(diff) * (diff < 0) + poly * (diff >= 0)) - l[true_label]) ** 2

        sorted_upper_bounds = u.sort(dim=0)
        max = sorted_upper_bounds[0][-1] if sorted_upper_bounds[0][-1] != u[true_label] else sorted_upper_bounds[0][-2]

        """Log loss """
        # loss = torch.log(max - l[true_label])
        """Linear loss """
        # loss = max - l[true_label]
        """Cross entropy loss"""
        L = torch.nn.CrossEntropyLoss()
        # L = torch.nn.NLLLoss()
        # m = torch.nn.LogSoftmax(dim=1)
        u[true_label] = l[true_label]
        softmax = torch.reshape(u, (1, 10))
        loss = L(softmax, torch.full((1,), true_label).type(torch.LongTensor))

        # Computing gradients and modifying slopes
        loss.backward()
        optimizer.step()

        # Adjusting optimizer's learning rate
        if prev_loss > loss:
            learning_rate = learning_rate / 0.9
        else:
            learning_rate = learning_rate * 0.9
        prev_loss = loss

        if DEBUG:
            print(number_runs, "time", "{0:.2f}".format(time.time() - start), "{0:.5f}".format(learning_rate), "loss",
                  loss.item(), "l[true_label]: ",
                  l[true_label].detach().numpy(), "max u: ", max.detach().numpy(), "params",
                  sum(p.numel() for p in slope_set if p.requires_grad))

        # Clipping to ensure soundness
        for s in range(len(slope_set)):
            slope_set[s].data = torch.clamp(slope_set[s].data, min=0, max=1)

        # Clear gradients
        optimizer.zero_grad()

        # In the case of convolutional networks, only the slopes of the last ReLU layer will be optimized
        if number_runs is 0 and freeze:
            layers = layers[2*threshold:]
            parameters = parameters[2*threshold:]
            slope_set = [slopes for slopes in slope_set[threshold:]]

        number_runs = number_runs + 1

    return 0


def freeze_all_but_last(threshold, num_relu, total_num_relu, l, u):
    """
Creates the set of slopes of current ReLU layer. The slopes will be optimized depending on the current ReLU layer
    :param threshold: -1 if all but last ReLU layer must be frozen, total_num_relu else
    :param num_relu: the current ReLU layer
    :param total_num_relu: total number of ReLU layers in the network
    :param l: The lower bounds of the zonotope
    :param u: The upper bounds of the zonotope
    :return: the set of slopes associated with the current ReLU layer
    """
    if num_relu - total_num_relu >= threshold:
        slopes = torch.tensor(u / (u - l), requires_grad=True)
    else:
        slopes = torch.tensor(u / (u - l), requires_grad=False)

    return slopes


def get_training_parameters(model, layers, total_num_relu):
    """
This method calculates the parameters used for training
    :param model: The string identifier for the model
    :param layers: The layers of the network
    :param total_num_relu: The number of relu layers in the network
    :return: The learning rate and threshold for freezing layers
    """
    threshold = -1 if (True in [isinstance(layer, torch.nn.Conv2d) for layer in layers]) else - total_num_relu
    if model == 'fc1':
        learning_rate = 0.01
    elif model == 'fc2':
        learning_rate = 0.01
    elif model == 'fc3':
        learning_rate = 0.01
    elif model == 'fc4':
        learning_rate = 0.01
    elif model == 'fc5':
        learning_rate = 0.01
    elif model == 'conv1':
        learning_rate = 0.01
    elif model == 'conv2':
        learning_rate = 0.01
    elif model == 'conv3':
        learning_rate = 0.01
    elif model == 'conv4':
        learning_rate = 0.01
    elif model == 'conv5':
        learning_rate = 0.01
    return learning_rate, threshold


def build_zonotope(inputs, eps):
    """
Builds the zonotope
    :param inputs: The image that should be verified
    :param eps: The magnitude of noise with which the image is perturbed
    :return: The zonotope abstracting the input image
    """
    noise = torch.ones(size=[len(inputs)]) * eps
    input_copy = inputs.clone()

    # Re centering centers and noise terms
    for i, pixel in enumerate(input_copy):
        if pixel + eps > 1:
            noise[i] = (1 - (input_copy[i] - eps)) / 2
            input_copy[i] = 1 - noise[i]
        if pixel - eps < 0:
            noise[i] = (input_copy[i] + eps) / 2
            input_copy[i] = noise[i]

    noise = torch.diag(noise)
    zonotope = torch.cat([input_copy.reshape(1, -1), noise], dim=0).T
    return zonotope


def affine_dense(zonotope, weight_matrix, bias):
    """
Pushes the zonotope through the dense layer associated with the weight matrix and bias arguments
    :param zonotope: The zonotope to be pushed
    :param weight_matrix: Weights of the layer
    :param bias: Bias of the layer
    :return: The transformed zonotope
    """
    result = torch.matmul(weight_matrix, zonotope)
    result[:, 0] = result[:, 0] + bias
    return result


def affine_conv(zonotope, layer, weight_matrix, bias, img_dim):
    """
Pushes the zonotope through the convolution layer associated with the weight matrix and bias arguments
    :param zonotope: The zonotope to be pushed
    :param layer: Weights of the layer
    :param weight_matrix:
    :param bias: Bias of the layer
    :param img_dim: The dimension of input
    :return:The transformed zonotope
    """
    zonotope = zonotope.reshape((layer.in_channels, img_dim, img_dim, -1))
    result = []

    for i in range(zonotope.shape[-1]):
        bias_ = bias if i == 0 else None
        temp = zonotope[:, :, :, i].reshape((1, layer.in_channels, img_dim, img_dim))
        temp = F.conv2d(temp, weight_matrix, bias=bias_,
                        stride=layer.stride, padding=layer.padding)
        result += [temp.reshape(1, -1)]

    zonotope = torch.stack(result).squeeze(1).T

    return zonotope


def relu(zonotope, l, u, slopes):
    """
Pushes the zonotope through the ReLU layer associated with the given set of slopes
    :param zonotope: The zonotope to be pushed
    :param l: The lower bounds of the zonotope
    :param u: The upper bounds of the zontope
    :param slopes: The set of slopes associated with each neuron of the layer
    :return: The transformed zonotope
    """
    added_dims = 0

    for i in range(zonotope.size(0)):
        if l[i] < 0 and u[i] > 0:
            added_dims += 1

    # Pre-allocating the new zonotope with adjusted dimensions
    result = torch.zeros((zonotope.size(0), zonotope.size(1) + added_dims))

    added = 0
    for i in range(zonotope.size(0)):
        if l[i] >= 0:
            result[i, :zonotope.size(1)] = zonotope[i].clone()
        elif u[i] <= 0:
            result[i, :zonotope.size(1)] = slopes[i] * torch.zeros(zonotope[i].size())
        else:
            opt_slope = u[i] / (u[i] - l[i])
            if slopes[i] <= opt_slope:
                temp = zonotope[i].clone()
                temp *= slopes[i]
                temp[0] += (u[i] / 2) * (1 - slopes[i])
                result[i, :zonotope.size(1)] = temp
                result[i, zonotope.size(1) + added] = ((u[i] / 2) * (1 - slopes[i])).reshape(1)
                added += 1
            else:
                temp = zonotope[i].clone()
                temp *= slopes[i]
                temp[0] -= l[i] * slopes[i] / 2
                result[i, :zonotope.size(1)] = temp
                result[i, zonotope.size(1) + added] = (-l[i] * slopes[i] / 2).reshape(1)
                added += 1
    return result


def compute_upper_lower_bounds(zonotope):
    """
Computes the lower and upper bounds of the zonotope
    :param zonotope: The zonotope
    :return: The lowe and upper bounds of the zonotope
    """
    (l, u) = (zonotope[:, 0], zonotope[:, 0])
    max = torch.sum(torch.abs(zonotope[:, 1:]), dim=1)
    (l, u) = (l - max, l + max)
    return l, u


def verify(zonotope, true_label):
    """
Checks if the zonotope is verified w.r.t the true label
    :param zonotope: The zonotope
    :param true_label: The label
    :return: 1 if the zonotope is verified, 0 else
    """
    l, u = compute_upper_lower_bounds(zonotope)
    threshold = l[true_label]
    sorted_upper_bounds = u.sort(dim=0)
    max = sorted_upper_bounds[0][-1] if sorted_upper_bounds[0][-1] != u[true_label] else sorted_upper_bounds[0][-2]

    return int(max <= threshold)


def main():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                        required=True,
                        help='Neural network to verify.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
    elif args.net == 'conv4':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    elif args.net == 'conv5':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label, args.net):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
