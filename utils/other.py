import random
import numpy
import torch
import collections


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = collections.OrderedDict()

    if type(array[0]) is torch.Tensor:
        #d["mean"] = tuple(numpy.array(torch.mean(torch.stack(array),axis=0).cpu()))
        #d["std"] = tuple(numpy.array(torch.std(torch.stack(array),axis=0).cpu()))
        #d["min"] = tuple(numpy.array(torch.min(torch.stack(array),axis=0).values.cpu()))
        #d["max"] = tuple(numpy.array(torch.max(torch.stack(array),axis=0).values.cpu()))
        d["mean"] = torch.mean(torch.stack(array),axis=0).cpu()
        d["std"] = torch.std(torch.stack(array),axis=0).cpu()
        d["min"] = torch.min(torch.stack(array),axis=0).values.cpu()
        d["max"] = torch.max(torch.stack(array),axis=0).values.cpu()
    else:
        d["mean"] = numpy.mean(array)
        d["std"] = numpy.std(array)
        d["min"] = numpy.min(array)
        d["max"] = numpy.max(array)
    return d
