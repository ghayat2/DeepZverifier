import argparse
import random
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from networks import FullyConnected, Conv, Normalization
import numpy as np
import torch.nn.functional as F
import time
import warnings

DEVICE = 'cpu'
INPUT_SIZE = 28
LEARNING_RATE = 0.00001


def analyze(net, inputs, eps, true_label):
    layers = [layer for layer in net.layers if not isinstance(layer, torch.nn.Flatten)]
    num_relu_layers = len([0 for i in range(len(layers)) if isinstance(layers[i], torch.nn.ReLU)])
    result = 0

    inputs = inputs.reshape(-1)
    initial_zonotope = build_zonotope(inputs, eps)
    slope_set = []
    number_runs = 0

    # Loss and Optimizer
    criterion = nn.L1Loss()

    """print(net)
    for param in list(net.parameters()):
        print("Param: ", param.size())"""

    end = time.time() + 60 * 3
    start = time.time()
    while time.time() < end:
        img_dim = INPUT_SIZE
        num_relu = 0

        for i in range(len(layers)):
            layer = layers[i]

            if isinstance(layer, Normalization):
                mean = layer.mean.reshape(-1)[0]
                sigma = layer.sigma.reshape(-1)[0]
                sdt = torch.eye(n=inputs.size(0)) * (1 / sigma)
                mean = torch.ones(size=inputs.size()) * (-mean / sigma)
                zonotope = affine_dense(initial_zonotope, sdt, mean)

            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
                weight_matrix = list(net.parameters())[i - 1].type(torch.float32)
                bias = list(net.parameters())[i].type(torch.float32)

                if isinstance(layer, torch.nn.Linear):
                    zonotope = affine_dense(zonotope, weight_matrix, bias)
                else:
                    zonotope = affine_conv(zonotope, layer, weight_matrix, bias, img_dim)
                    img_dim = img_dim // layer.stride[0]

            if isinstance(layer, torch.nn.ReLU):
                (l, u) = compute_upper_lower_bounds(zonotope)
                if number_runs is 0:
                    slopes = torch.tensor(u / (u - l), requires_grad=True)
                    slope_set.append(slopes)

                """print("The slope used in relu layers are: ")
                for slopes in slope_set:
                    print(slopes)"""

                zonotope = relu(zonotope, l, u, slopes=slope_set[num_relu])
                num_relu = num_relu + 1

        # Early stop
        result = verify(zonotope, true_label)
        if result:
            print(time.time() - start)
            return result

        # define optimizer and weights to train
        optimizer = optim.Adam(slope_set, lr=0.01)

        # calculate loss
        (l, u) = compute_upper_lower_bounds(zonotope)
        """actual_error = l[true_label]
        sorted_upper_bounds = u.sort(dim=0)
        optimal_error = torch.zeros(actual_error.size())
        max = sorted_upper_bounds[0][-1] if sorted_upper_bounds[0][-1] != u[true_label] else sorted_upper_bounds[0][-2]
        print("l[true_label]: ", actual_error, "max u: ", max)
        current_loss = criterion(max, optimal_error)
        current_loss.backward()
        # print("Current loss is :", current_loss)"""

        # diff = torch.sum(torch.exp(diff) * (diff < 0) + poly * (diff >= 0)) - l[true_label]
        sorted_upper_bounds = u.sort(dim=0)
        max = sorted_upper_bounds[0][-1] if sorted_upper_bounds[0][-1] != u[true_label] else sorted_upper_bounds[0][-2]

        # loss = torch.log(max - l[true_label])**3
        loss = max - l[true_label]

        optimizer.zero_grad()
        loss.backward()

        # print("loss", loss.item(),  "l[true_label]: ", l[true_label].detach().numpy(), "max u: ", max.detach().numpy())


        #print("Size of slope_set", len(slope_set[0]))
        # step
        # print("Before optimizer", slope_set[0].grad)
        t1 = slope_set[0].grad
        optimizer.step()
        #print("After optimizer", slope_set[0].grad)
        t2 = slope_set[0].grad

        # print("Changed: ", False in [t1[i] == t2[i] for i in range(len(t1))])

        # clip and repeat
        for s in range(len(slope_set)):
            slope_set[s].data = slope_set[s].data.clamp(0, 1)

        # clear gradients
        optimizer.zero_grad()
        number_runs = number_runs + 1

    return result


def build_zonotope(inputs, eps):
    noise = torch.ones(size=[len(inputs)]) * eps
    input_copy = inputs.clone()

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
    result = torch.matmul(weight_matrix, zonotope)
    result[:, 0] = result[:, 0] + bias
    return result


def affine_conv(zonotope, layer, weight_matrix, bias, img_dim):
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
    result = []
    added = 0

    for i in range(len(zonotope)):
        if l[i] >= 0:
            result.append(zonotope[i])
        elif u[i] <= 0:
            result.append(torch.zeros(zonotope[i].size()))
        else:
            opt_slope = u[i] / (u[i] - l[i])
            if slopes[i] <= opt_slope:
                temp = zonotope[i].clone()
                temp *= slopes[i]
                temp[0] += (u[i] / 2) * (1 - slopes[i])
                result.append(torch.cat(
                    [temp, torch.cat([torch.zeros(added), ((u[i] / 2) * (1 - slopes[i])).reshape(1)], dim=0)],
                    axis=0))
                # result.append(t)
                added += 1
            else:
                temp = zonotope[i].clone()
                temp *= slopes[i]
                temp[0] -= l[i] * slopes[i] / 2
                result.append(torch.cat([temp, torch.cat([torch.zeros(added), (-l[i] * slopes[i] / 2).reshape(1)], dim=0)],
                              axis=0))

                # result.append(t)
                added += 1

    target_size = zonotope.size()[1] + added
    for i in range(len(result)):
        result[i] = torch.cat([result[i], torch.zeros(target_size - len(result[i]))])

    result = torch.stack(result).reshape(zonotope.size()[0], -1)
    return result


def compute_upper_lower_bounds(zonotope):
    (l, u) = (zonotope[:, 0], zonotope[:, 0])
    max = torch.sum(torch.abs(zonotope[:, 1:]), dim=1)
    (l, u) = (l - max, l + max)
    return l, u


def verify(zonotope, true_label):
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

    t = time.time()
    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')

    # print("Execution time: ", time.time() - t, "s")


if __name__ == '__main__':
    main()
