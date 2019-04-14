from policy.model import train

kwargs = {
    'n_epochs': 3,
    'batch_size': 64,
}

train(**kwargs)