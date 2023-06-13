import copy

def add_noise(x):
    pass

def add_head_mark(x):
    pass

def derive(x):
    x_copy = copy.deepcopy(x)
    enc_x = add_noise(x)
    dec_x = add_head_mark(x)
    return enc_x, dec_x, x
