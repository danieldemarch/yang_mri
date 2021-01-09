
def get_environment(args, split):
    if args.env == 'rec_v1':
        from .rec_env_v1 import RecEnv
        env = RecEnv(args, split)
    elif args.env == 'rec_v2':
        from .rec_env_v2 import RecEnv
        env = RecEnv(args, split)
    else:
        raise NotImplementedError()

    return env