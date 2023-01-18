import argparse
import math
import random
import os

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from results_json import ResultsJSON

import mnist_dataset
import uci_datasets
from difflogic import LogicLayer,GroupSum, PackBitsTensor, CompiledLogicNet
from difflogic.functional import bin_op_s
import difflogic_cuda

torch.set_num_threads(1)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}

class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda.backward_w(x, a, b, grad_y)
        return grad_x, None, None, grad_w, None, None, None

def load_dataset(args):
    validation_loader = None
    if args.dataset == 'adult':
        train_set = uci_datasets.AdultDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = uci_datasets.AdultDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    elif args.dataset == 'breast_cancer':
        train_set = uci_datasets.BreastCancerDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = uci_datasets.BreastCancerDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    elif args.dataset.startswith('monk'):
        style = int(args.dataset[4])
        train_set = uci_datasets.MONKsDataset('./data-uci', style, split='train', download=True, with_val=False)
        test_set = uci_datasets.MONKsDataset('./data-uci', style, split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    elif args.dataset in ['mnist', 'mnist20x20']:
        train_set = mnist_dataset.MNIST('./data-mnist', train=True, download=True, remove_border=args.dataset == 'mnist20x20')
        test_set = mnist_dataset.MNIST('./data-mnist', train=False, remove_border=args.dataset == 'mnist20x20')

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
    elif 'cifar-10' in args.dataset:
        transform = {
            'cifar-10-3-thresholds': lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0),
            'cifar-10-31-thresholds': lambda x: torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=0),
        }[args.dataset]
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(transform),
        ])
        train_set = torchvision.datasets.CIFAR10('./data-cifar', train=True, download=True, transform=transforms)
        test_set = torchvision.datasets.CIFAR10('./data-cifar', train=False, transform=transforms)

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    else:
        raise NotImplementedError(f'The data set {args.dataset} is not supported!')

    return train_loader, validation_loader, test_loader


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


def input_dim_of_dataset(dataset):
    return {
        'adult': 116,
        'breast_cancer': 51,
        'monk1': 17,
        'monk2': 17,
        'monk3': 17,
        'mnist': 784,
        'mnist20x20': 400,
        'cifar-10-3-thresholds': 3 * 32 * 32 * 3,
        'cifar-10-31-thresholds': 3 * 32 * 32 * 31,
    }[dataset]


def num_classes_of_dataset(dataset):
    return {
        'adult': 2,
        'breast_cancer': 2,
        'monk1': 2,
        'monk2': 2,
        'monk3': 2,
        'mnist': 10,
        'mnist20x20': 10,
        'cifar-10-3-thresholds': 10,
        'cifar-10-31-thresholds': 10,
    }[dataset]


def get_model(args):
    llkw = dict(grad_factor=args.grad_factor, connections=args.connections, temp = 0.1)

    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)

    logic_layers = []

    arch = args.architecture
    k = args.num_neurons
    l = args.num_layers

    ####################################################################################################################

    if arch == 'randomly_connected':
        logic_layers=[]
        logic_layers.append(torch.nn.Flatten())
        #logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
        k = [784, 576*10, 400*10, 256*10, 144*10, 64*10, 8*10, 4*10, 2*10, 1*10]
        r = [10,1,1,1,1,1,1,1,1]
        sq_size = [28,24, 20, 16, 12, 8, 4, 2, 1]
        for idx in range(l):
            logic_layers.append(LogicLayer(in_dim=k[idx], out_dim=k[idx+1], redundance = r[idx],sq_size = sq_size[idx], **llkw))

        model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, args.tau)
        )

    ####################################################################################################################

    else:
        raise NotImplementedError(arch)

    ####################################################################################################################

    total_num_neurons = sum(map(lambda x: x.num_neurons, logic_layers[1:-1]))
    print(f'total_num_neurons={total_num_neurons}')
    total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:-1]))
    print(f'total_num_weights={total_num_weights}')
    if args.experiment_id is not None:
        results.store_results({
            'total_num_neurons': total_num_neurons,
            'total_num_weights': total_num_weights,
        })

    model = model.to('cuda')

    print(model)
    if args.experiment_id is not None:
        results.store_results({'model_str': str(model)})

    loss_fn = torch.nn.CrossEntropyLoss().to("cuda")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer


def train(model, x, y, loss_fn, optimizer):
    x = model(x)
    loss = loss_fn(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def eval(model, loader, mode):
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        res = np.mean(
            [
                (model(x.to('cuda').round()).argmax(-1) == y.to('cuda')).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()


def packbits_eval(model, loader):
    orig_mode = model.training
    with torch.no_grad():
        model.eval()
        res = np.mean(
            [
                (model(PackBitsTensor(x.to('cuda').round().bool())).argmax(-1) == y.to('cuda')).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()

class LogicLayer(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """
    def __init__(
            self,
            in_dim: int = 784,
            out_dim: int = 576,
            redundance: int = 1,
            sq_size: int = 1,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
            temp: float = 1.0,
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        """
        super().__init__()
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16, device=device))
        self.weights_connections = torch.nn.parameter.Parameter(torch.randn(in_dim, 2, out_dim, device=device))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.temp = temp

        """
        The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only 
        available for device='cuda'. The `python` implementation exists for 2 reasons:
        1. To provide an easy-to-understand implementation of differentiable logic gate networks 
        2. To provide a CPU implementation of differentiable logic gate networks 
        """
        self.implementation = implementation
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'python'], self.implementation

        self.connections = connections
        assert self.connections in ['random', 'unique', 'learned'], self.connections
        self.indices = self.get_connections(self.connections, device)
        hold = int(in_dim/(sq_size*sq_size))
        print(hold)
        single_out_dim = out_dim//(redundance * hold)
        self.mask = torch.zeros((sq_size,sq_size, hold, out_dim), device=device).detach()
        i=0
        j=0
        r = 0
        h = 0
        for k in range(out_dim):
            for h in range(hold):
                self.mask[i:i+5, j:j+5, h, k] = 1
            r+=1
            if r >= redundance:
                r =0
                i += 1
            if i+5>sq_size:
                i=0
                j+=1
        #self.mask = self.mask.reshape(-1, single_out_dim)
        #self.mask = self.mask.repeat(hold,hold*redundance)
        self.mask = self.mask.reshape(-1, out_dim)
        print(self.mask.shape)
        print(self.weights_connections[:,0,:].shape)
        #with torch.no_grad():
        #    self.weights_connections[:,0,:] *= self.mask
        #    self.weights_connections[:,1,:] *= self.mask

        if self.implementation == 'cuda':
            """
            Defining additional indices for improving the efficiency of the backward of the CUDA implementation.
            """
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices[0].cpu().numpy()
            indices_1_np = self.indices[1].cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), device=device, dtype=torch.int64)
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist], dtype=torch.int64, device=device)

        self.num_neurons = out_dim
        self.num_weights = out_dim
        
    def get_connections(self, connections, device='cuda'):
        #assert self.out_dim * 2 >= self.in_dim, 'The number of neurons ({}) must not be smaller than half of the ' \
        #                                        'number of inputs ({}) because otherwise not all inputs could be ' \
        #                                        'used or considered.'.format(self.out_dim, self.in_dim)
        if connections == 'random':
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        elif connections == 'unique':
            return get_unique_connections(self.in_dim, self.out_dim, device)
        elif connections == 'learned':
            a, b = self.learned_connections_eval()
            return a, b
        else:
            raise ValueError(connections)

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            assert not self.training, 'PackBitsTensor is not supported for the differentiable training mode.'
            assert self.device == 'cuda', 'PackBitsTensor is only supported for CUDA, not for {}. ' \
                                          'If you want fast inference on CPU, please use CompiledDiffLogicModel.' \
                                          ''.format(self.device)

        else:
            if self.grad_factor != 1.:
                x = GradFactor.apply(x, self.grad_factor)
        if self.connections == 'learned':
            if self.training:
                return self.forward_python_modified(x)
            else:
                self.indices = self.learned_connections_eval()
        
        if self.implementation == 'cuda':
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x)
        elif self.implementation == 'python':
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)
        
    def learned_connections_eval(self):
        c = torch.max(self.weights_connections, dim=0).indices
        a, b = c[0,:], c[1,:]
        return a, b

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            print(self.indices[0].dtype, self.indices[1].dtype)
            self.indices = self.indices[0].long(), self.indices[1].long()
            print(self.indices[0].dtype, self.indices[1].dtype)

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = bin_op_s(a, b, torch.nn.functional.softmax(self.weights/self.temp, dim=-1))
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
            x = bin_op_s(a, b, weights)
        return x
        
    def forward_python_modified(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            self.indices = self.indices[0].long(), self.indices[1].long()
        #with torch.no_grad():
        #    self.weights_connections[:,0,:] *= self.mask
        #    self.weights_connections[:,1,:] *= self.mask
        x = x.type(torch.float)
        weighting_func = torch.nn.functional.softmax(self.weights_connections/self.temp, dim = 0)
        a = torch.einsum('ij,ki->kj', weighting_func[:,0,:], x)
        b = torch.einsum('ij,ki->kj', weighting_func[:,1,:], x)
        x = bin_op_s(a, b, torch.nn.functional.softmax(self.weights/self.temp, dim=-1))
        return x

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)
        x = x.contiguous()

        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices

        if self.training:
            w = torch.nn.functional.softmax(self.weights / self.temp, dim=-1).to(x.dtype)
            return LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                return LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)

    def forward_cuda_eval(self, x: PackBitsTensor):
        """
        WARNING: this is an in-place operation.

        :param x:
        :return:
        """
        assert not self.training
        assert isinstance(x, PackBitsTensor)
        assert x.t.shape[0] == self.in_dim, (x.t.shape, self.in_dim)

        a, b = self.indices
        w = self.weights.argmax(-1).to(torch.uint8)
        x.t = difflogic_cuda.eval(x.t, a, b, w)

        return x

    def extra_repr(self):
        return '{}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval')
        
class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """
    def __init__(self, k: int, tau: float = 1., device='cuda'):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            return x.group_sum(self.k)

        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau

    def extra_repr(self):
        return 'k={}, tau={}'.format(self.k, self.tau)


if __name__ == '__main__':

    ####################################################################################################################

    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')

    parser.add_argument('-eid', '--experiment_id', type=int, default=None)

    parser.add_argument('--dataset', type=str, choices=[
        'adult', 'breast_cancer',
        'monk1', 'monk2', 'monk3',
        'mnist', 'mnist20x20',
        'cifar-10-3-thresholds',
        'cifar-10-31-thresholds',
    ], required=True, help='the dataset to use')
    parser.add_argument('--tau', '-t', type=float, default=10, help='the softmax temperature tau')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 32)')

    parser.add_argument('--implementation', type=str, default='cuda', choices=['cuda', 'python'],
                        help='`cuda` is the fast CUDA implementation and `python` is simpler but much slower '
                        'implementation intended for helping with the understanding.')

    parser.add_argument('--packbits_eval', action='store_true', help='Use the PackBitsTensor implementation for an '
                                                                     'additional eval step.')
    parser.add_argument('--compile_model', action='store_true', help='Compile the final model with C for CPU.')

    parser.add_argument('--num-iterations', '-ni', type=int, default=100_000, help='Number of iterations (default: 100_000)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=2_000, help='Evaluation frequency (default: 2_000)')

    parser.add_argument('--valid-set-size', '-vss', type=float, default=0., help='Fraction of the train set used for validation (default: 0.)')
    parser.add_argument('--extensive-eval', action='store_true', help='Additional evaluation (incl. valid set eval).')

    parser.add_argument('--connections', type=str, default='learned', choices=['random', 'unique'])
    parser.add_argument('--architecture', '-a', type=str, default='randomly_connected')
    parser.add_argument('--num_neurons', '-k', type=int)
    parser.add_argument('--num_layers', '-l', type=int)

    parser.add_argument('--grad-factor', type=float, default=1.)

    args = parser.parse_args()

    ####################################################################################################################

    print(vars(args))

    assert args.num_iterations % args.eval_freq == 0, (
        f'iteration count ({args.num_iterations}) has to be divisible by evaluation frequency ({args.eval_freq})'
    )

    if args.experiment_id is not None:
        assert 520_000 <= args.experiment_id < 530_000, args.experiment_id
        results = ResultsJSON(eid=args.experiment_id, path='./results/')
        results.store_args(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, validation_loader, test_loader = load_dataset(args)
    def hook_fn(m, i, o):
        print(o.shape)
    for num_layers in range(1, 7):
        args.num_layers = num_layers
        model, loss_fn, optim = get_model(args)
        if num_layers > 1:
            for this_layer_idx in range(1, num_layers):
                model[this_layer_idx].weights.requires_grad_(False)
                model[this_layer_idx].weights_connections.requires_grad_(False)
                model[this_layer_idx].weights.copy_(prev_state_dict[str(this_layer_idx)+'.weights'])
                model[this_layer_idx].weights_connections.copy_(prev_state_dict[str(this_layer_idx)+'.weights_connections'])
        ####################################################################################################################

        best_acc = 0
        temp = 0.1
        '''
        for layer in model:
            if isinstance(layer, LogicLayer):
                layer.temp = temp
            if isinstance(layer, GroupSum):
                layer.tau = temp
        '''
        ascending = False
        
        for i, (x, y) in tqdm(
                enumerate(load_n(train_loader, args.num_iterations)),
                desc='iteration',
                total=args.num_iterations,
        ):
            x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
            y = y.to('cuda')

            loss = train(model, x, y, loss_fn, optim)
            #print(loss)

            if (i+1) % args.eval_freq == 0:
                '''
                if temp > 1:
                    ascending = False
                if temp < 0.1:
                    ascending = True
                '''
                if ascending:
                    temp *= 2
                else:
                    temp *= 0.9
                for layer in model:
                    if isinstance(layer, LogicLayer):
                        layer.temp = temp
                    #if isinstance(layer, GroupSum):
                    #    layer.tau = temp
                
                if args.extensive_eval:
                    train_accuracy_train_mode = eval(model, train_loader, mode=True)
                    valid_accuracy_eval_mode = eval(model, validation_loader, mode=False)
                    valid_accuracy_train_mode = eval(model, validation_loader, mode=True)
                else:
                    train_accuracy_train_mode = -1
                    valid_accuracy_eval_mode = -1
                    valid_accuracy_train_mode = -1
                train_accuracy_eval_mode = eval(model, train_loader, mode=False)
                test_accuracy_eval_mode = eval(model, test_loader, mode=False)
                test_accuracy_train_mode = eval(model, test_loader, mode=True)

                r = {
                    'train_acc_eval_mode': train_accuracy_eval_mode,
                    'train_acc_train_mode': train_accuracy_train_mode,
                    'valid_acc_eval_mode': valid_accuracy_eval_mode,
                    'valid_acc_train_mode': valid_accuracy_train_mode,
                    'test_acc_eval_mode': test_accuracy_eval_mode,
                    'test_acc_train_mode': test_accuracy_train_mode,
                    'temperature layers': temp,
                }

                if args.packbits_eval:
                    r['train_acc_eval'] = packbits_eval(model, train_loader)
                    r['valid_acc_eval'] = packbits_eval(model, train_loader)
                    r['test_acc_eval'] = packbits_eval(model, test_loader)

                if args.experiment_id is not None:
                    results.store_results(r)
                else:
                    print(r)

                if valid_accuracy_eval_mode > best_acc:
                    best_acc = valid_accuracy_eval_mode
                    if args.experiment_id is not None:
                        results.store_final_results(r)
                    else:
                        print('IS THE BEST UNTIL NOW.')

                if args.experiment_id is not None:
                    results.save()
        prev_state_dict = model.state_dict()

    ####################################################################################################################

    if args.compile_model:
        print('\n' + '='*80)
        print(' Converting the model to C code and compiling it...')
        print('='*80)

        for opt_level in range(4):

            for num_bits in [
                #8,
                #16,
                #32,
                64
            ]:
                os.makedirs('lib', exist_ok=True)
                save_lib_path = 'lib/{:08d}_{}.so'.format(
                    args.experiment_id if args.experiment_id is not None else 0, num_bits
                )

                compiled_model = CompiledLogicNet(
                    model=model,
                    num_bits=num_bits,
                    cpu_compiler='gcc',
                    # cpu_compiler='clang',
                    verbose=True,
                )

                compiled_model.compile(
                    opt_level=1 if args.num_layers * args.num_neurons < 50_000 else 0,
                    save_lib_path=save_lib_path,
                    verbose=True
                )

                correct, total = 0, 0
                with torch.no_grad():
                    for (data, labels) in torch.utils.data.DataLoader(test_loader.dataset, batch_size=int(1e6), shuffle=False):
                        data = torch.nn.Flatten()(data).bool().numpy()

                        output = compiled_model(data, verbose=True)

                        correct += (output.argmax(-1) == labels).float().sum()
                        total += output.shape[0]

                acc3 = correct / total
                print('COMPILED MODEL', num_bits, acc3)

