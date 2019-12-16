import argparse
import torch
from networks import FullyConnected, Conv, Normalization
import numpy as np
import torch.nn.functional as F

DEVICE = 'cpu'
INPUT_SIZE = 28


def analyze(net, inputs, eps, true_label):
    img_dim = INPUT_SIZE

    inputs = inputs.reshape(-1).double()
    noise =  (torch.eye(len(inputs)) * eps).double()
    zonotope = torch.cat((inputs.reshape(1, -1), noise), dim=0).T.double()


    for i in range(len(net.layers)):
        layer = net.layers[i]
        if isinstance(layer, Normalization):
            mean = layer.mean.double().item()
            sigma = layer.sigma.double().item()
            sdt = torch.eye(len(inputs)) * (1 / sigma)
            mean = torch.ones(len(inputs)) * (-mean / sigma)
            zonotope = affine_dense(zonotope, sdt, mean)

        if isinstance(layer, torch.nn.Linear):
            weight_matrix = list(net.parameters())[i - 2].data.double()
            bias = list(net.parameters())[i - 1].data.double()
            zonotope = affine_dense(zonotope, weight_matrix, bias)

        if isinstance(layer, torch.nn.Conv2d):
            weight_matrix = list(net.parameters())[i - 1].data.double()
            bias = list(net.parameters())[i].data.double()
            zonotope, img_dim = affine_conv(zonotope, layer, weight_matrix, bias, img_dim)

        if isinstance(layer, torch.nn.ReLU):
            zonotope = relu(zonotope)

    result = verify(zonotope, true_label)
    return result


def affine_dense(zonotope, weight_matrix, bias):
    result = torch.mm(weight_matrix, zonotope)
    result[:, 0] = result[:, 0] + bias
    return result


def affine_conv(zonotope, layer, weight_matrix, bias, img_dim):
    zonotope = zonotope.reshape((layer.in_channels, img_dim, img_dim, -1))

    result = []
    for i in range(zonotope.shape[-1]):
        if i != 0:
            bias = None

        temp = zonotope[:, :, :, i].reshape((1, layer.in_channels, img_dim, img_dim))
        temp = F.conv2d(temp, weight_matrix, bias=bias,
                        stride=layer.stride, padding=layer.padding)
        result.append(temp.reshape(-1))

    zonotope = torch.stack(result).T
    img_dim = img_dim // layer.stride[0]
    return zonotope, img_dim


def relu(zonotope):
    (l, u) = compute_upper_lower_bounds(zonotope)
    result = []
    added = 0
    for i in range(len(zonotope)):
        if l[i] >= 0:
            result.append(zonotope[i])
        elif u[i] <= 0:
            result.append(torch.zeros(zonotope[1].size()))
        else:
            slope = u[i] / (u[i] - l[i])
            temp = zonotope[i]
            temp *= slope
            temp[0] -= (slope * l[i]) / 2
            result.append(torch.cat([temp, torch.zeros(added), (-(slope * l[i]) / 2).reshape(1)], 0))
            added += 1

    target_size = zonotope.size()[1] + added
    for i in range(len(result)):
        result[i] = torch.cat([result[i], torch.zeros((target_size - len(result[i])))])
    result = torch.cat(result).reshape(zonotope.size()[0], -1)

    return result


def compute_upper_lower_bounds(zonotope):
    (l, u) = (zonotope[:, 0], zonotope[:, 0])
    max = torch.sum(torch.abs(zonotope[:, 1:]), dim=1)
    (l, u) = (l - max, l + max)
    return l, u


def verify(zonotope, true_label):
    l, u = compute_upper_lower_bounds(zonotope)
    print("l", l)
    print("u", u)
    threshold = l[true_label]
    sorted_upper_bounds = sorted(u)
    max = sorted_upper_bounds[-1] if sorted_upper_bounds[-1] != u[true_label] else sorted_upper_bounds[-2]
    return int(max <= threshold)


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

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()