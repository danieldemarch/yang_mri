
def get_agent(args, model, env):
    if args.policy == 'ppo':
        from .ppo import PPOPolicy
        agent = PPOPolicy(args, model, env)
    else:
        raise NotImplementedError()

    return agent