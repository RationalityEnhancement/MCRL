import numpy as np
import scipy.stats
import itertools as it

def option_util(x,sigma):
    return sigma*scipy.stats.norm.pdf(x/sigma) - np.abs(x)*scipy.stats.norm.cdf(-np.abs(x)/sigma)

def all_option_insts(g_acts,g_probs,n_obs):
    insts = [[]]
    n_remaining_obs = n_obs

    vals, inverse, count = np.unique(g_probs, return_inverse=True,
                                  return_counts=True)
    rows, cols = np.where(inverse == np.arange(len(vals))[:, np.newaxis])
    _, inverse_rows = np.unique(rows, return_index=True)
    res = np.split(cols, inverse_rows[1:])

    for i in range(len(res)):
        new_insts = []

        n_new_nodes = len(res[-i-1])
        if n_new_nodes < n_remaining_obs:
            n_remaining_obs -= n_new_nodes
        else:
            n_new_nodes = n_remaining_obs
            n_remaining_obs = 0

        for new_nodes in it.permutations(res[-i-1],n_new_nodes):
            for inst in insts:
                new_insts.append(inst + list(np.array(g_acts)[list(new_nodes)]))
        insts = new_insts
        if n_remaining_obs == 0:
            break

    return insts

def get_all_options(env):
    actions = list(env.actions()) #list of all the available actions
    g_counts = [0,]*env.gambles #number of available cells for each gamble
    g_acts = [] #the available actions in each gamble
    g_probs = [] #the probability of the unobserved outcomes

    options = [] #list of all available options
    option_utils = [] #list of the utility of each option
    # the instantiation is only useful if there are two outcomes are equally likely
    option_insts = dict() #list of all possible option instantiations

    for i in range(env.gambles):
        probs = []
        acts = [] #available actions

        for j in range(env.outcomes):
            a = i*env.outcomes+j
            if a in actions:
                probs.append(env.dist[j])
                acts.append(a)
                g_counts[i] += 1

        g_probs.append(probs)
        g_acts.append(acts)

        for j in range(g_counts[i]):
            options.append((i,j+1))

    max_obs = np.max(env.mus)

    for option in options:
        gamble, obs = option
        sig = np.sqrt(np.sum(np.sort(g_probs[gamble])[::-1][:obs]**2*env.reward.sigma**2))
        option_utils.append((option_util(env.mus[gamble]-max_obs,sig)+obs*env.cost)/obs)
        option_insts[option] = all_option_insts(g_acts[gamble],g_probs[gamble],obs)

    #single click options
    sc_opt = (-1,1)
    options.append(sc_opt)
    option_utils.append(-np.inf)
    option_insts[sc_opt] = [[a] for a in env.actions()]
    n_available_clicks = len(option_insts[sc_opt])

    #end click options
    end_opt = (-99,1)
    options.append(end_opt)
    option_utils.append(0)
    option_insts[end_opt] = [[env.term_action]]

    return options, option_insts, np.array(option_utils), n_available_clicks, g_acts, g_probs, g_counts

def pick_option_moves(env):
    options, option_insts, option_utils, nac, g_acts, g_probs, g_counts = get_all_options(env)

    #c is for chosen
    cgamble, cobs = options[np.random.choice(np.arange(
            len(options))[option_utils == np.max(option_utils)])]
    if cgamble == -99:
        return np.array([env.term_action]), (cgamble,cobs)

    cgamble_ps = np.array(g_probs[cgamble])
    cgamble_nodes = np.array(g_acts[cgamble])
    b = np.random.random(cgamble_nodes.size)

    return cgamble_nodes[np.lexsort((b,cgamble_ps))[::-1][:cobs]], (cgamble,cobs)

def run_dc(env):
    all_acts = []
    rews = []
    opts = []
    while not(env._state is env.term_state):
        acts, opt = pick_option_moves(env)
        opts.append(opt)
        # print(acts)
        for a in acts:
            f,r,d,c = env._step(a)
            rews.append(r)
        all_acts += list(acts)
    # env.reset()
    return {'util': np.sum(rews), 'actions': all_acts, 'options':opts,
            'observations': len(all_acts) - 1, 'ground_truth': env.ground_truth}
