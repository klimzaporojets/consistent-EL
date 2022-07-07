import logging

import torch.nn as nn

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


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
        logging.info('forward {}: {} / {} = {}'
                     .format(self.label, norm_outputs, norm_inputs, norm_outputs / norm_inputs))
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
