import os
import sys

import torch


def do(from_path, to_path):
    ckp = torch.load(from_path)
    keys = list(ckp.keys())
    for k in keys:
        v = ckp.pop(k)
        if k.startswith('user_encoder'):
            ckp[k[13:]] = v

    if not to_path.endswith('.pt'):
        name = os.path.split(from_path)[-1]
        to_path = os.path.join(to_path, name)
    torch.save(ckp, to_path)


if __name__ == '__main__':
    f, t = sys.argv[1], sys.argv[2]
    do(f, t)
