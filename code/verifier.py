import argparse
import random
from itertools import product

import torch
from networks import FullyConnected, Conv, Normalization
import numpy as np
import torch.nn.functional as F
import time

DEVICE = 'cpu'
INPUT_SIZE = 28
BRANCHING_FACTOR = 5


def analyze(net, inputs, eps, true_label):
    num_relu_layers = len([0 for i in range(len(net.layers)) if isinstance(net.layers[i], torch.nn.ReLU)])
    slopes_to_try = construct_set(BRANCHING_FACTOR, num_relu_layers)
    result = 0

    inputs = inputs.numpy().reshape(-1)

    calculation_set = construct_calculation_set(net.layers)

    while not result and len(slopes_to_try) > 0:

        img_dim = INPUT_SIZE
        indices = random.choice(tuple(slopes_to_try))
        slopes_to_try.discard(indices)
        zonotope = build_zonotope(inputs, eps)
        num_relu = 0
        
        for i in range(len(net.layers)):
            layer = net.layers[i]

            if isinstance(layer, Normalization):

                if calculation_set[i][0] is None:
                    # calculate new zonotope
                    mean = np.array(layer.mean).reshape(-1)[0]
                    sigma = np.array(layer.sigma).reshape(-1)[0]
                    sdt = np.diag(np.ones(shape=len(inputs)) * (1 / sigma))
                    mean = np.ones(shape=len(inputs)) * (-mean / sigma)
                    zonotope = affine_dense(zonotope, sdt, mean)
                    calculation_set[i][0] = zonotope
                else:
                    zonotope = calculation_set[i][0]

            if isinstance(layer, torch.nn.Flatten):

                store_index = calc_store_index(indices, num_relu)

                if calculation_set[i][store_index] is None:
                    calculation_set[i][store_index] = zonotope
                else:
                    zonotope = calculation_set[i][store_index]

            if isinstance(layer, torch.nn.Linear):

                store_index = calc_store_index(indices, num_relu)

                if calculation_set[i][store_index] is None:
                    prev_relu_index = indices[num_relu - 1]
                    weight_matrix = list(net.parameters())[i - 2].data.numpy()
                    bias = list(net.parameters())[i - 1].data.numpy()
                    zonotope = affine_dense(zonotope, weight_matrix, bias)
                    calculation_set[i][store_index] = zonotope
                else:
                    zonotope = calculation_set[i][store_index]

            if isinstance(layer, torch.nn.Conv2d):

                store_index = calc_store_index(indices, num_relu)

                if calculation_set[i][store_index] is None:
                    weight_matrix = list(net.parameters())[i - 1].data.numpy().astype(float)
                    bias = list(net.parameters())[i].data.numpy().astype(float)
                    zonotope = affine_conv(zonotope, layer, weight_matrix, bias, img_dim)
                    calculation_set[i][store_index] = zonotope
                else:
                    zonotope = calculation_set[i][store_index]
                
                img_dim = img_dim // layer.stride[0]

            if isinstance(layer, torch.nn.ReLU):

                store_index = calc_store_index(indices, num_relu + 1)

                if calculation_set[i][store_index] is None:
                    (l, u) = compute_upper_lower_bounds(zonotope)
                    slopes = u / (u - l)
                    if indices[num_relu] == 0:
                        zonotope = relu(zonotope, l, u, slopes=np.zeros(slopes.shape))
                    elif indices[num_relu] == 1:
                        zonotope = relu(zonotope, l, u, slopes=0.5 * slopes)
                    elif indices[num_relu] == 2:
                        zonotope = relu(zonotope, l, u, slopes=slopes)
                    elif indices[num_relu] == 3:
                        zonotope = relu(zonotope, l, u, slopes=slopes + (1 - slopes) / 2)
                    elif indices[num_relu] == 4:
                        zonotope = relu(zonotope, l, u, slopes=np.ones(slopes.shape))

                    calculation_set[i][store_index] = zonotope
                else:
                    zonotope = calculation_set[i][store_index]

                num_relu += 1

        result = verify(zonotope, true_label)

    return result

def calc_store_index(indices, num_relu):
    if num_relu is 0:
        return 0;
    else:
        slope_index = indices[num_relu - 1] + 5 * calc_store_index(indices, num_relu - 1)
        return slope_index


def construct_set(branching_factor, n_relu_layer):
    return set(product(range(branching_factor), repeat=n_relu_layer))

def construct_calculation_set(layers):
    calculation_set = [];
    for i in range(len(layers)):
        calculation_set.append([])

    for i in range(len(layers)):
        layer = layers[i]
        if isinstance(layer, Normalization):
            calculation_set[i].append(None)
        elif isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Flatten):
            for j in range(len(calculation_set[i-1])):
                calculation_set[i].append(None)
        elif isinstance(layer, torch.nn.ReLU):
            for j in range(len(calculation_set[i-1])):
                for k in range(BRANCHING_FACTOR):
                    calculation_set[i].append(None)

    return calculation_set



def build_zonotope(inputs, eps):
    noise = np.ones(shape=(len(inputs))) * eps
    input_copy = np.copy(inputs)

    for i, pixel in enumerate(input_copy):
        if pixel + eps > 1:
            noise[i] = (1 - (input_copy[i] - eps)) / 2
            input_copy[i] = 1 - noise[i]
        if pixel - eps < 0:
            noise[i] = (input_copy[i] + eps) / 2
            input_copy[i] = noise[i]

    noise = np.diag(noise)
    zonotope = np.concatenate((input_copy.reshape(1, -1), noise), axis=0).T
    return zonotope


def affine_dense(zonotope, weight_matrix, bias):
    result = np.matmul(weight_matrix, zonotope)
    result[:, 0] = result[:, 0] + bias
    return result


def affine_conv(zonotope, layer, weight_matrix, bias, img_dim):
    zonotope = zonotope.reshape((layer.in_channels, img_dim, img_dim, -1))
    result = []

    for i in range(zonotope.shape[-1]):
        if i == 0:
            bias_ = torch.from_numpy(bias)
        else:
            bias_ = None

        temp = zonotope[:, :, :, i].reshape((1, layer.in_channels, img_dim, img_dim))
        temp = F.conv2d(torch.from_numpy(temp), torch.from_numpy(weight_matrix), bias=bias_,
                        stride=layer.stride, padding=layer.padding)
        result.append(temp.numpy().reshape(-1))

    zonotope = np.array(result).T
    return zonotope


def relu(zonotope, l, u, slopes):
    result = []
    added = 0

    for i in range(len(zonotope)):
        if l[i] >= 0:
            result.append(zonotope[i])
        elif u[i] <= 0:
            result.append(np.zeros(shape=np.shape(zonotope[i])))
        else:
            opt_slope = u[i] / (u[i] - l[i])
            slope = slopes[i]

            if slope <= opt_slope:
                temp = np.array(zonotope[i])
                temp *= slope
                temp[0] += (u[i] / 2) * (1 - slope)
                result.append(np.append(np.concatenate([temp, np.zeros(shape=added)]), (u[i] / 2) * (1 - slope)))
                added += 1
            else:
                temp = np.array(zonotope[i])
                temp *= slope
                temp[0] -= l[i] * slope / 2
                result.append(np.append(np.concatenate([temp, np.zeros(shape=added)]), -l[i] * slope / 2))
                added += 1

    target_size = np.shape(zonotope)[1] + added
    for i in range(len(result)):
        result[i] = np.concatenate([result[i], np.zeros(shape=(target_size - len(result[i])))])

    result = np.array(result).reshape((np.shape(zonotope)[0], -1))
    return result


def compute_upper_lower_bounds(zonotope):
    (l, u) = (zonotope[:, 0], zonotope[:, 0])
    max = np.array(np.sum(np.abs(zonotope[:, 1:]), axis=1))
    (l, u) = (l - max, l + max)
    return l, u


def verify(zonotope, true_label):
    l, u = compute_upper_lower_bounds(zonotope)
    threshold = l[true_label]
    sorted_upper_bounds = sorted(u)
    max = sorted_upper_bounds[-1] if sorted_upper_bounds[-1] != u[true_label] else sorted_upper_bounds[-2]
    return int(max <= threshold)


def debug(zonotope, weight_matrix, bias):
    print("############## DEBUG ##############")
    print("Zonotope", np.shape(zonotope))
    print("Weight matrix", np.shape(weight_matrix))
    print("Bias", np.shape(bias))
    print("###############################################")


def main():
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

    print("Execution time: ", time.time() - t, "s")


if __name__ == '__main__':
    main()
