
def get_dataset(split, args):
    if args.dataset == 'kmnist':
        from .kmnist import Dataset
        dataset = Dataset(split, args.batch_size)
    elif args.dataset == 'kcifar10':
        from .kcifar10 import Dataset
        dataset = Dataset(split, args.batch_size)
    elif args.dataset == 'cifar10':
        from .cifar10 import Dataset
        dataset = Dataset(split, args.batch_size)
    elif args.dataset == 'knee':
        from .knee import Dataset
        dataset = Dataset(split, args.batch_size, args.max_acquisition, args.center_acquisition)
    else:
        raise NotImplementedError()

    return dataset