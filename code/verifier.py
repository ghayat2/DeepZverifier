import argparse
import torch
from networks import FullyConnected, Conv, Normalization
import numpy as np
import torch.nn.functional as F
import time

DEVICE = 'cpu'
INPUT_SIZE = 28


def analyze(net, inputs, eps, true_label):
    img_dim = INPUT_SIZE
    inputs = inputs.numpy().reshape(-1)
    zonotopes = [build_zonotope(inputs, eps)]

    for i in range(len(net.layers)):

        layer = net.layers[i]

        if isinstance(layer, Normalization):
            mean = np.array(layer.mean).reshape(-1)[0]
            sigma = np.array(layer.sigma).reshape(-1)[0]

            sdt = np.diag(np.ones(shape=len(inputs)) * (1 / sigma))
            mean = np.ones(shape=len(inputs)) * (-mean / sigma)
            zonotopes = [affine_dense(zonotopes[0], sdt, mean)]

        if isinstance(layer, torch.nn.Linear):
            weight_matrix = list(net.parameters())[i - 2].data.numpy()
            bias = list(net.parameters())[i - 1].data.numpy()
            zonotopes = [affine_dense(zonotope, weight_matrix, bias) for zonotope in zonotopes]

        if isinstance(layer, torch.nn.Conv2d):
            weight_matrix = list(net.parameters())[i - 1].data.numpy().astype(float)
            bias = list(net.parameters())[i].data.numpy().astype(float)
            zonotopes = [affine_conv(zonotope, layer, weight_matrix, bias, img_dim) for zonotope in zonotopes]
            img_dim = img_dim // layer.stride[0]

        if isinstance(layer, torch.nn.ReLU):
            res = []
            for zonotope in zonotopes:
                (l, u) = compute_upper_lower_bounds(zonotope)
                slopes = u / (u - l)

                res.append(relu(zonotope, l, u, slopes=np.zeros(slopes.shape)))
                res.append(relu(zonotope, l, u, slopes=0.5*slopes))
                res.append(relu(zonotope, l, u, slopes=slopes))
                res.append(relu(zonotope, l, u, slopes=slopes + (1-slopes)/2))
                res.append(relu(zonotope, l, u, slopes=np.ones(slopes.shape)))

            zonotopes.clear()
            zonotopes += res.copy()

    result = np.array([verify(zonotope, true_label) for zonotope in zonotopes])
    ones = len([1 for r in result if r == 1])
    print(len(result)-ones, " zeros, ", ones, " ones")
    return any(result > 0)


def build_zonotope(inputs, eps):
    noise = np.ones(shape=(len(inputs))) * eps

    for i, pixel in enumerate(inputs):
        if pixel + eps > 1:
            noise[i] = (1 - (inputs[i] - eps)) / 2
            inputs[i] = 1 - noise[i]
        if pixel - eps < 0:
            noise[i] = (inputs[i] + eps) / 2
            inputs[i] = noise[i]

    noise = np.diag(noise)
    zonotope = np.concatenate((inputs.reshape(1, -1), noise), axis=0).T
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
                temp[0] += (u[i] / 2)*(1 - slope)
                result.append(np.append(np.concatenate([temp, np.zeros(shape=added)]), (u[i] / 2)*(1 - slope)))
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
    #print(id(zonotope))
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
    t1 = time.clock()
    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')

    t2 = time.clock()
    print("Execution time: ", t2-t1, "s")


if __name__ == '__main__':
    main()
