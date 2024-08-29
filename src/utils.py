import sys
import time
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import top_k_accuracy_score, f1_score
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    alexnet,
)

from src.resnet import resnet18 as cresnet18


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed=seed)


def set_model_state(model, parameters):
    state = model.state_dict()
    counted_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = torch.tensor(
                parameters[counted_params : param.size().numel() + counted_params]
            ).reshape(param.size())
            counted_params += param.size().numel()

    model.load_state_dict(state)

    return model


def get_model_params(model):
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(torch.flatten(param).cpu().detach().numpy())

    return np.concatenate(params)


def init_pop_in_block(NP, codebook, params):
    BD = len(codebook)
    min_blocks, max_blocks = np.zeros(BD), np.zeros(BD)

    for idx, block in codebook.items():
        min_blocks[idx], max_blocks[idx] = np.min(params[block]), np.max(params[block])

    return np.random.uniform(min_blocks, max_blocks, size=(NP, BD))


def f1score_func(model, data_loader, num_classes, device, mode="val"):
    model.eval()
    fitness = 0
    all_outputs, all_labels = (
        torch.Tensor([]).to(device),
        torch.Tensor([]).to(device),
    )
    t1 = time.time()
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            all_outputs = torch.cat((all_outputs, preds))
            all_labels = torch.cat((all_labels, labels))
            # )
            # if mode == "val":
            #     break
    t2 = time.time()
    fitness = f1_score(
        all_outputs.cpu().detach(),
        all_labels.cpu().detach(),
        average="macro",
        labels=np.arange(num_classes),
    )
    t3 = time.time()
    # print(t2 - t1, t3 - t2)
    return fitness


def get_network(net, dataset, model_path):
    model = None
    if net == "resnet18" and dataset == "imagenet":
        model = resnet18(weights="DEFAULT")
    elif net == "resnet34" and dataset == "imagenet":
        model = resnet34(weights="DEFAULT")
    elif net == "resnet50" and dataset == "imagenet":
        model = resnet50(weights="DEFAULT")
    elif net == "resnet101" and dataset == "imagenet":
        model = resnet101(weights="DEFAULT")
    elif net == "resnet152" and dataset == "imagenet":
        model = resnet152(weights="DEFAULT")
    elif net == "alexnet" and dataset == "imagenet":
        model = alexnet(weights="DEFAULT")
    elif net == "resnet18" and dataset == "cifar10":
        model = cresnet18(num_classes=10)
        model.load_state_dict(torch.load(model_path))
    elif net == "resnet18" and dataset == "cifar100":
        model = cresnet18(num_classes=100)
        model.load_state_dict(torch.load(model_path))

    else:
        print("the network name you have entered is not supported yet")
        sys.exit()

    return model


# Node of a Huffman Tree
class Node:
    def __init__(self, probability, index, left=None, right=None):
        # probability of the symbol
        self.probability = probability
        # the symbol
        self.index = index
        # the left node
        self.left = left
        # the right node
        self.right = right
        # the tree direction (0 or 1)
        self.code = ""


class HuffmanEncode:
    def __init__(self, bits=5):
        self.symbols, self.codes = {}, {}
        self.initial_bits = bits

    """ Calculates frequency of every index in data"""

    def frequency(self, data):
        indices, frequencies = np.unique(data, return_counts=True)
        return indices, frequencies

    def codify(self, node, value=""):
        # a huffman code for current node
        newValue = value + str(node.code)

        if node.left:
            self.codify(node.left, newValue)
        if node.right:
            self.codify(node.right, newValue)

        if not node.left and not node.right:
            self.codes[node.index] = newValue
        return self.codes

    @classmethod
    def encode(cls, data, bits=5):
        huffman = cls(bits=bits)
        symbols, frequencies = huffman.frequency(data)
        nodes = []

        # converting symbols and probabilities into huffman tree nodes
        for s, f in zip(symbols, frequencies):
            nodes.append(Node(f, s))

        while len(nodes) > 1:
            # sorting all the nodes in ascending order based on their probability
            nodes = sorted(nodes, key=lambda x: x.probability)
            # for node in nodes:
            #      print(node.index, node.prob)

            # picking two smallest nodes
            right = nodes[0]
            left = nodes[1]

            left.code = 0
            right.code = 1

            # combining the 2 smallest nodes to create new node
            new = Node(
                left.probability + right.probability,
                left.index + right.index,
                left,
                right,
            )

            nodes.remove(left)
            nodes.remove(right)
            nodes.append(new)

        huffmanEncoding = huffman.codify(nodes[0])
        # print("symbols with codes", huffmanEncoding)
        tot_size, avg_bits = huffman.get_gain(data, huffmanEncoding)
        # encoded = huffman.get_encoded(data, huffmanEncoding)
        return tot_size, avg_bits

    def get_gain(self, data, coding):
        # total bit space to store the data before compression
        n_data = len(data)
        print(n_data)
        before = n_data * self.initial_bits
        after = 0
        symbols = coding.keys()
        # use indexing to iterate over symbols
        i = 0
        for symbol in symbols:
            count = np.count_nonzero(data == symbol)
            # calculating how many bit is required for that symbol in total
            after += count * len(coding[symbol])
            # log.debug(f"  Symbol: {symbol} | count: {count:.0f} | coding length: {len(coding[symbol])}")
        print(
            "  Space usage before huffman encoding for {:.0f} values (in bits): {:.0f}".format(
                n_data, before
            )
        )
        print(
            "  Space usage after huffman encoding for {:.0f} values (in bits): {:.0f}".format(
                n_data, after
            )
        )
        print("  Average bits: {:.1f}".format(after / n_data))
        return after, after / n_data


def report_huffman_encoding(x_path, params_size):

    # x_path = f"out/nsga2_resnet18_cifar10_mergev1_hard/codebooks/merged_codebook_bmax_{B_max}.pkl"
    print(x_path)
    if os.path.exists(x_path):
        with open(
            x_path,
            "rb",
        ) as f:
            xopt_codebook = pickle.load(f)

    block_sizes = []
    block_var_bits = []
    histogram_block_codebook_size_nonempty = {}

    counter = 0
    for i, block in xopt_codebook.items():
        n = len(block)
        block_sizes.append(n)
        block_var_bits.append(np.ceil(np.log2(n)))
        histogram_block_codebook_size_nonempty[counter] = n
        counter += 1

    # plt.figure(figsize=(1, 1))
    plt.bar(
        histogram_block_codebook_size_nonempty.keys(),
        histogram_block_codebook_size_nonempty.values(),
        color="green",
    )
    plt.yscale("log")  # Change y-axis to logarithmic scale
    plt.ylabel("block size")
    plt.xlabel("block index")
    plt.show()

    block_var_bits = []
    sum_var_bits = []
    arg_sorted_block_sizes = np.argsort(block_sizes)
    sorted_block_sizes = np.sort(block_sizes)
    for i in range(len(sorted_block_sizes)):
        bi = np.ceil(np.log2(i + 1))
        if bi == 0:
            bi = 1.0
        block_var_bits.append(bi)

        size_i = sorted_block_sizes[len(sorted_block_sizes) - i - 1]
        sum_var_bits.append(bi * (size_i + 1))

    data = np.zeros(params_size)

    for i, block in xopt_codebook.items():
        data[block] = np.full(len(block), i)

    after_hf, after_t = HuffmanEncode.encode(
        data, bits=np.ceil(np.log2(len(xopt_codebook)))
    )

    before_t = np.ceil(np.log2(len(xopt_codebook)))

    after_cr = (32 * params_size) / (after_hf + 32 * len(xopt_codebook))
    before_cr = (32 * params_size) / (
        np.ceil(np.log2(len(xopt_codebook))) * params_size + 32 * len(xopt_codebook)
    )

    return before_cr, before_t, after_cr, after_t
