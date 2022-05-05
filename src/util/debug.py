import gc
import math

import torch
import torch.nn as nn


def print_allocated_tensors():
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    print(type(obj), obj.size())
                    total_size += obj.numel()
        except:
            pass
    print("total size:", total_size * 4, "=", total_size / 1024 / 1024 * 4)
    print("allocated: ", )


def masked_inspect(name, tensor, mask):
    unmasked_minimum = tensor.min().item()
    unmasked_maximum = tensor.max().item()
    # print('tensor:', tensor.size())
    # print('mask:', mask.size())
    mask = mask.expand_as(tensor)
    masked_minimum = (tensor * mask + (1.0 - mask) * unmasked_maximum).min().item()
    masked_maximum = (tensor * mask + (1.0 - mask) * unmasked_minimum).max().item()

    N = mask.sum().item()
    average = (tensor * mask).sum().item() / mask.sum().item()
    x = (tensor - average) * mask
    stdev = math.sqrt((x * x).sum().item() / N)
    print('INSPECT\t{}: min={} max={} avg={} std={}'.format(name, masked_minimum, masked_maximum, average, stdev))


class Wrapper1(nn.Module):

    def __init__(self, label, module, dim_output=None):
        super(Wrapper1, self).__init__()
        self.label = label
        self.module = module
        self.dim_output = module.dim_output if dim_output is None else dim_output

    def forward(self, inputs):
        outputs = self.module(inputs)
        norm_inputs = inputs.norm().item()
        norm_outputs = outputs.norm().item()
        print("forward {}: {} / {} = {}".format(self.label, norm_outputs, norm_inputs, norm_outputs / norm_inputs))
        return outputs


import sys
# https://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python
# http://web.archive.org/web/20141016185743/https://mail.python.org/pipermail/python-list/2007-May/460639.html
class Tee(object):

    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def close(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        self.close()
