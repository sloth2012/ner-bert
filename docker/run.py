# coding: utf8

from bailian_nlp.web.server import main

if __name__ == '__main__':
    import urllib3

    urllib3.disable_warnings()

    import torch
    import time
    import torch.multiprocessing as mp

    torch.manual_seed(int(time.time()))
    mp.set_start_method('spawn', force=True)
    main()
