# DeepZverifier
The goal of the project is to design a precise and scalable automated verifier for proving the robustness of
fully connected and convolutional neural networks with rectified linear activations (ReLU) against
adversarial attacks. We will consider adversarial attacks based on the L∞ -norm perturbations. A network is
vulnerable to this type of attack if there exists a misclassified image inside an L∞ -norm based ε -ball around
the original image.


## Folder structure
In the directory `code` you can find 2 files. 
File `networks.py` contains encoding of fully connected and convolutional neural network architectures as PyTorch classes.
The architectures extend `nn.Module` object and consist of standard PyTorch layers (`Linear`, `Flatten`, `ReLU`, `Conv2d`). Please note that first layer of each network performs normalization of the input image.
File `verifier.py` contains a template of verifier. Loading of the stored networks and test cases is already implemented in `main` function.

In folder `mnist_nets` you can find 10 neural networks (5 fully connected and 5 convolutional). These networks are loaded using PyTorch in `verifier.py`.
In folder `test_cases` you can find 10 subfolders. Each subfolder is associated with one of the networks, using the same name. In a subfolder corresponding to a network, you can find 2 test cases for this network. 

## Setup instructions

We recommend you to install Python virtual environment to ensure dependencies are same as the ones we will use for evaluation.
To evaluate your solution, we are going to use Python 3.6.9.
You can create virtual environment and install the dependencies using the following commands:

```bash
$ virtualenv venv --python=python3.6
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Running the verifier

Run the verifier from `code` directory using the command:

```bash
$ python verifier.py --net {net} --spec ../test_cases/{net}/img{test_idx}_{eps}.txt
```

In this command, `{net}` is equal to one of the following values (each representing one of the networks we want to verify): `fc1, fc2, fc3, fc4, fc5, conv1, conv2, conv3, conv4, conv5`.
`test_idx` is an integer representing index of the test case, while `eps` is perturbation that verifier should certify in this test case.

To test the verifier, you can run for example:

```bash
$ python verifier.py --net fc1 --spec ../test_cases/fc1/img0_0.06000.txt
```

To evaluate the verifier on all networks and sample test cases, we provide the evaluation script.
You can run this script using the following commands:

```bash
chmod +x evaluate
./evaluate
```

For more information about the implementation of the verifier, please see the report.pdf file.
