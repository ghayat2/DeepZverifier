import argparse
import torch
import torch.optim as optim
from networks import FullyConnected, Conv, Normalization
import torch.nn.functional as F
import time
import warnings

DEVICE = 'cpu'
INPUT_SIZE = 28
LEARNING_RATE = 0.00001


def analyze(net, inputs, eps, true_label):
    layers = [layer for layer in net.layers if not isinstance(layer, torch.nn.Flatten)]
    total_num_relu = sum([1 for layer in net.layers if isinstance(layer, torch.nn.ReLU)])

    inputs = inputs.reshape(-1)
    initial_zonotope = build_zonotope(inputs, eps)

    slope_set = []
    number_runs = 0
    result = 0

    end = time.time() + 60 * 2 * 12  # server est approx 12x plus rapide que mon ordi
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
                    u = u.detach()
                    l = l.detach()

                    if num_relu - total_num_relu >= -7:
                        slopes = torch.tensor(u / (u - l), requires_grad=True)
                        print("Optimizing")
                    else:
                        slopes = torch.tensor(u / (u - l), requires_grad=False)
                        print("Not optimizing")
                    # slopes = torch.tensor(u / (u - l), requires_grad=True)

                    slope_set.append(slopes)

                zonotope = relu(zonotope, l, u, slopes=slope_set[num_relu])
                num_relu = num_relu + 1

        # Early stop
        result = verify(zonotope, true_label)
        if result:
            print(time.time() - start)
            return result

        # define optimizer and weights to train
        optimizer = optim.Adam(slope_set, lr=0.003)

        # calculate loss
        (l, u) = compute_upper_lower_bounds(zonotope)
        diff = u - l[true_label]
        diff[true_label] = 0
        poly = diff + 1
        loss = (torch.sum(torch.exp(diff) * (diff < 0) + poly * (diff >= 0)) - l[true_label]) ** 2
        sorted_upper_bounds = u.sort(dim=0)
        max = sorted_upper_bounds[0][-1] if sorted_upper_bounds[0][-1] != u[true_label] else sorted_upper_bounds[0][-2]
        # loss = torch.log(max - l[true_label])
        # loss = max - l[true_label]
        # print("Number of params to optimize: ", sum([slopes.size()[0] for slopes in slope_set]) )
        params = sum(p.numel() for p in slope_set if p.requires_grad)
        print("PARAMS ", params)
        print("Calling backwards..")
        t1 = time.time()
        loss.backward()
        print("Done in ", time.time() - t1, "s")
        optimizer.step()

        print("loss", loss.item(), "l[true_label]: ", l[true_label].detach().numpy(), "max u: ", max.detach().numpy())

        for s in range(len(slope_set)):
            slope_set[s].data = torch.clamp(slope_set[s].data, min=0)

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
    added_dims = 0
    for i in range(zonotope.size(0)):
        if l[i] < 0 and u[i] > 0:
            added_dims += 1

    result = torch.zeros((zonotope.size(0), zonotope.size(1) + added_dims))

    added = 0
    for i in range(zonotope.size(0)):
        if l[i] >= 0:
            result[i, :zonotope.size(1)] = zonotope[i].clone()
        elif u[i] <= 0:
            continue
            # result[i] = torch.zeros(zonotope[i].size())
        else:
            if slopes[i] == 0:
                temp = zonotope[i].clone()
                temp *= slopes[i]
                temp[0] += u[i] / 2
                result[i, :zonotope.size(1)] = temp
                result[i, zonotope.size(1) + added] = (u[i] / 2).reshape(1)
                added += 1
            elif 0 < slopes[i] <= 1:
                temp = zonotope[i].clone()
                temp *= slopes[i]
                temp[0] += 0.5 * max(- slopes[i] * l[i], u[i] * (1 - slopes[i]))
                result[i, :zonotope.size(1)] = temp
                result[i, zonotope.size(1) + added] = (0.5 * max(- slopes[i] * l[i], u[i] * (1 - slopes[i]))).reshape(1)
                added += 1
            else:
                temp = zonotope[i].clone()
                temp *= slopes[i]
                temp[0] += (0.5 * u[i] * (slopes[i] - 1) - 0.5 * slopes[i] * l[i])
                result[i, :zonotope.size(1)] = temp
                result[i, zonotope.size(1) + added] = (-0.5 * u[i] * (slopes[i] - 1) - 0.5 * slopes[i] * l[i]).reshape(
                    1)
                added += 1
    return result


def relu_tmp(zonotope, l, u, slopes):
    added_dims = 0
    for i in range(zonotope.size(0)):
        if l[i] < 0 and u[i] > 0:
            added_dims += 1

    result = torch.zeros((zonotope.size(0), zonotope.size(1) + added_dims))

    added = 0
    for i in range(zonotope.size(0)):
        if l[i] >= 0:
            result[i, :zonotope.size(1)] = zonotope[i].clone()
        elif u[i] <= 0:
            continue
            # result[i] = torch.zeros(zonotope[i].size())
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
