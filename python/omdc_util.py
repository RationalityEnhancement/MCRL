import numpy as np
import scipy.stats

def option_util(x,sigma):
    return sigma*scipy.stats.norm.pdf(x/sigma) - np.abs(x)*scipy.stats.norm.cdf(-np.abs(x)/sigma)

def get_all_options(env):
    actions = list(env.actions())
    g_counts = [0,]*env.gambles
    g_obs = []
    g_unobs = []
    g_probs = []

    options = []
    option_utils = []

    for i in range(env.gambles):
        probs = []
        acts = []

        obs = env.reward.mu
        
        for j in range(env.outcomes):
            a = i*env.outcomes+j
            if a in actions:
                probs.append(env.dist[j])
                acts.append(a)
                g_counts[i] += 1
            else:
                obs += env.dist[j]*(env._state[a]-env.reward.mu)

        g_obs.append(obs)
        g_probs.append(probs)
        g_unobs.append(acts)

        for j in range(g_counts[i]):
            options.append((i,j+1))

    max_obs = np.max(g_obs)
    for option in options:
        path, obs = option
        sig = np.sqrt(np.sum(np.sort(g_probs[path])[::-1][:obs]**2*env.reward.sigma))
        option_utils.append((option_util(g_obs[path]-max_obs,sig)+obs*env.cost)/obs)
        
    options.append([-1,1])
    option_utils.append(0)
    
    return options, option_utils, g_unobs, g_probs, g_obs, g_counts

def pick_option_moves(env):
    options, option_utils, g_unobs, g_probs, g_obs, g_counts = get_all_options(env)

    #c is for chosen
    cgamble, cobs = options[np.random.choice(np.arange(len(options))[option_utils == np.max(option_utils)])]
    if cgamble == -1:
        return np.array([env.term_action])
    
    cgamble_ps = np.array(g_probs[cgamble])
    cgamble_nodes = np.array(g_unobs[cgamble])
    b = np.random.random(cgamble_nodes.size)

    return cgamble_nodes[np.lexsort((b,cgamble_ps))[::-1][:cobs]]

def run_dc(env):
    all_acts = []
    rews = []
    while not(env._state is env.term_state):
        acts = pick_option_moves(env)
        # print(acts)
        for a in acts:
            f,r,d,c = env._step(a)
            rews.append(r)
        all_acts += list(acts)
    # env.reset()
    return {'util': np.sum(rews), 'actions': all_acts,
            'observations': len(all_acts) - 1, 'ground_truth': env.ground_truth}