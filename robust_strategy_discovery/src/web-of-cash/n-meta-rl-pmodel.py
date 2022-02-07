'''
code modified from https://github.com/awjuliani/Meta-RL
'''
# In[ ]:


import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import scipy.signal
from helper1 import *
from random import choice
from time import sleep
from time import time
import glob
import shutil
from mouselab import MouselabEnv
from distributions import Categorical, Normal
import pandas as pd
from posterior import get_train_envs
from sklearn.manifold import TSNE
import seaborn as sns




# In[ ]:

def processState(instate):
    mask = np.ones(len(instate))
    state = np.ones(len(instate))
    for i in range(0, len(instate)):
        if hasattr(instate[i], 'sample'):
            mask[i] = -1
            state[i] = 0
        else:
            mask[i] = 1
            state[i] = instate[i]

    mask = mask[:] * 60


    state_arr = [
    [state[3],state[4],state[5]],
    [state[2],state[6],state[10]],
    [state[1],state[14],state[27]],
    ]

    mask_arr = [
    [mask[3],mask[4],mask[5]],
    [mask[2],mask[6],mask[10]],
    [mask[1],mask[14],mask[27]],
    ]

    out = np.stack((state_arr,mask_arr), axis=0)
    return out

# In[ ]:

def performAction(env,action):
    if action==0:
        return env.step(action)
    else:
        if action==1:
            ac_set=[1]

        if action==2:
            ac_set=[14]

        if action==3:
            ac_set=[27]

        if action==4:
            ac_set=[2,15,28]

        if action==5:
            ac_set =[6,19,32]

        if action==6:
            ac_set=[10,23,36]

        if action==7:
            ac_set=[3,7,11,16,20,24,29,33,37]

        if action==8:
            ac_set=[4,8,12,17,21,25,30,34,38]

        if action==9:
            ac_set=[5,9,13,18,22,26,31,35,39]


        for i in ac_set:
            s1, r, d, obs = env.step(i)
        if(obs==True):
            r=env.repeat_cost
        else:
            r=env.cost

        return s1,r,d,obs


# In[ ]:


class AC_Network():
    def __init__(self, a_size, scope, trainer):
        with tf.variable_scope(scope):

            self.state = tf.placeholder(shape=[None,2,3,3], dtype=tf.float32)

            self.state_NHWC = tf.transpose(self.state,[0,2,3,1])

            self.conv1 = slim.conv2d(inputs=self.state_NHWC, num_outputs=32, kernel_size=[1, 1], stride=[
                                     1, 1], padding='VALID', biases_initializer=None)

            self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=164, kernel_size=[1, 1], stride=[
                                     1, 1], padding='VALID', biases_initializer=None)

            self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=128, kernel_size=[1, 3], stride=[
                                     1, 1], padding='VALID',biases_initializer=None)

            self.conv3 = slim.conv2d(inputs=self.conv3, num_outputs=95, kernel_size=[3, 1], stride=[
                                    1, 1], padding='VALID',biases_initializer=None)




            self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.timestep = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.prev_actions_onehot = tf.one_hot(self.prev_actions, a_size, dtype=tf.float32)

            self.hidden = tf.concat([slim.flatten(self.conv3), self.prev_rewards, self.prev_actions_onehot], 1)


            lstm_cell = tf.contrib.rnn.BasicLSTMCell(126, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(self.hidden, [0])
            step_size = tf.shape(self.prev_rewards)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 126])
            self.rnn_out=rnn_out


            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(
                                                   0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(
                                                  1.0),
                                              biases_initializer=None)

            self.pmodel = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(
                                                  1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.target_pmodel = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(
                    shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(
                    self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * \
                    tf.reduce_sum(tf.square(self.target_v -
                                            tf.reshape(self.value, [-1])))
                self.entropy = - \
                    tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
                self.policy_loss = - \
                    tf.reduce_sum(
                        tf.log(self.responsible_outputs + 1e-7) * self.advantages)

                self.pmodel_loss = 0.5 * \
                    tf.reduce_sum(tf.square(self.target_pmodel -
                                            tf.reshape(self.pmodel, [-1])))


                self.loss = .1 * self.value_loss + self.policy_loss - self.entropy * .1+  self.pmodel_loss

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(
                    self.gradients, 50)

                # Apply local gradients to global network
                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(
                    zip(grads, global_vars))




# In[ ]:


class Worker():
    def __init__(self, game, name, a_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter('tfb')

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        timesteps = rollout[:, 3]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        values = rollout[:, 5]

        self.pr = prev_rewards
        self.pa = prev_actions
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * \
            self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        pmodel = rollout[:,6]

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.state: np.stack(states, axis=0),
                     self.local_AC.prev_rewards: np.vstack(prev_rewards),
                     self.local_AC.prev_actions: prev_actions,
                     self.local_AC.actions: actions,
                     self.local_AC.timestep: np.vstack(timesteps),
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1],
                     self.local_AC.target_pmodel: pmodel,
                     }
        v_l, t_l,p_l,pmodel_l ,e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.pmodel_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout),t_l / len(rollout), p_l / len(rollout),pmodel_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver, train):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("start worker:", self.name)

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                r = 0
                a = n_actions
                t = 0

                env = self.env[np.random.randint(0, len(self.env))]
                sP = env.reset()
                s = processState(sP)
                rnn_state = self.local_AC.state_init

                while episode_step_count < n_actions:
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state_new = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                                        feed_dict={
                        self.local_AC.state: [s],
                        self.local_AC.prev_rewards: [[r]],
                        self.local_AC.timestep: [[t]],
                        self.local_AC.prev_actions: [a],
                        self.local_AC.state_in[0]: rnn_state[0],
                        self.local_AC.state_in[1]: rnn_state[1]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    rnn_state = rnn_state_new

                    s1P, r, d, obs = performAction(env,a)
                    s1 = processState(s1P)

                    episode_buffer.append([s, a, r, t, d, v[0, 0],env.process_model*5])
                    episode_values.append(v[0, 0])
                    episode_reward += r
                    total_steps += 1
                    t += 1
                    episode_step_count += 1
                    s = s1

                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0 and train == True:
                    v_l, t_l,p_l,pmodel_l ,e_l, g_n, v_n = self.train(
                        episode_buffer, sess, gamma, 0.0)

                if episode_count % 2000 == 0 and episode_count != 0:
                    if(save_model == True):
                        if(episode_count % 2000 == 0):
                            saver.save(sess, model_path + str(episode_count) + '.cptk')

                    if self.name == 'worker_0':
                        mean_reward = np.mean(self.episode_rewards[-2000:])
                        print(episode_count,mean_reward)

                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Reward',
                                          simple_value=float(mean_reward))
                        summary.value.add(
                                tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(
                                tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))

                        summary.value.add(
                                tag='Losses/Pmodel Loss', simple_value=float(pmodel_l))

                        summary.value.add(
                                tag='Losses/Total Loss', simple_value=float(t_l))


                        self.summary_writer.add_summary(summary, episode_count)

                        self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)

                    if(episode_count >= num_episodes):

                        coord.request_stop()

                episode_count += 1



# In[ ]:


gamma = .95  # discount rate for advantage estimation and reward discounting
num_episodes = 800000
train = True
model_path = "./models/temp/"  # The path to save our model to.
load_model =True
save_model = True
n_actions = 10
max_epLength = n_actions
a_size = n_actions
env_array,pos = get_train_envs(1)
type=1


# In[ ]:
if load_model==False:

    try:
        shutil.rmtree('./tfb')
    except OSError:
        print ("no dir")

    files = glob.glob(model_path+'*')
    for f in files:
        os.remove(f)

tf.reset_default_graph()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(
        0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=2e-4)
    # Generate global network
    master_network = AC_Network(a_size, 'global', None)

    # num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    num_workers = 3
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(env_array, i, a_size, trainer,
                              model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=10)
# In[ ]:
if type==0:




    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.all_model_checkpoint_paths[5])
        else:
            sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            def worker_work(): return worker.work(gamma, sess, coord, saver, train)
            thread = threading.Thread(target=(worker_work))
            thread.start()
            worker_threads.append(thread)
        coord.join(worker_threads)


# In[ ]:




model_path = "./models/meta-pmodel/"



df = pd.DataFrame(columns=['i', 'return','env_type', 'actions', 'ground_truth','pmodels'])
with tf.Session() as sess:
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.all_model_checkpoint_paths[6])
    #saver.restore(sess,ckpt.model_checkpoint_path )
    if True:
        if True:
            resum = 0
            tsne_input = np.empty((0,126))
            tsne_env_num = np.empty((0))
            tsne_pmodel = np.empty((0))
            for i in range(len(env_array)):

                episode_reward = 0
                episode_step_count = 0
                d = False
                r = 0
                a = n_actions
                t = 0

                actions = []
                pmodels=[]
                env = env_array[i]
                sP = env.reset()
                s = processState(sP)
                rnn_state = master_network.state_init



                while episode_step_count < max_epLength:

                    # Take an action using probabilities from policy network output.
                    hidden_vec,a_dist, v, rnn_state_new,pmodel = sess.run([master_network.rnn_out,master_network.policy, master_network.value,master_network.state_out, master_network.pmodel],
                                                        feed_dict={
                        master_network.state: [s],
                        master_network.prev_rewards: [[r]],
                        master_network.timestep: [[t]],
                        master_network.prev_actions: [a],
                        master_network.state_in[0]: rnn_state[0],
                        master_network.state_in[1]: rnn_state[1]})

                    if(episode_step_count==4):

                        tsne_input = np.append(tsne_input,hidden_vec,axis=0)
                        tsne_env_num = np.append(tsne_env_num,[i])
                        tsne_pmodel = np.append(tsne_pmodel,[list(pmodel[0])[0]],axis=0)

                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    rnn_state = rnn_state_new

                    s1P, r, d, obs = performAction(env,a)
                    s1 = processState(s1P)

                    if(obs == True):

                        s = s1
                        t += 1
                        episode_step_count += 1

                        continue

                    episode_reward += r
                    actions.append(a)
                    pmodel = list(pmodel[0])[0]
                    pmodels.append(pmodel)
                    episode_step_count += 1
                    s = s1
                    t += 1

                    if d == True:
                        break

                if(d == False):
                    a = 0
                    print("hello")
                    actions.append(a)
                    s1, r, d, obs = env.step(a)
                    episode_reward += r

                if(i<500):
                    etype = 1
                else:
                    etype = 2

                df.loc[i] = [i, episode_reward,etype, actions, env.ground_truth,pmodels]
                #print(i, pmodels)
                resum += episode_reward


print(resum / len(env_array))
#df.to_csv('csv/temp.csv')




# In[ ]:

tsne = TSNE(n_components=2)
tsne_output = tsne.fit_transform(tsne_input)
# In[ ]:
tsne_output.shape
# In[ ]:
tsne_plabel=[]
for i in tsne_pmodel:
    if(i>2):
        t=0
    elif(i<-2):
        t=1
    else:
        t=2
    tsne_plabel.append(t)
# In[ ]:
breaks = [0, 0, 0, 274, 340, 297, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 323, 0, 0, 379, 0, 0, 376, 0, 0, 345, 0, 0,
 347, 0, 0, 313, 0]

total=0
end_points=[]
for m in range(36):

    total = total+breaks[m]
    end=total
    end_points.append(end)
a = list(set(end_points))
a.sort()
end_points=a[1:]

t_type=[]
for i in tsne_env_num:
    for j in range(8):
        if i<end_points[j]:
            type = j
            break
    t_type.append(type)


t_type[0]=0
t_type[1]=1
t_type[2]=2
t_type[3]=3
t_type[4]=4
t_type[5]=5
t_type[5]=6
t_type[7]=7

# In[ ]:
original_plabel=[]
for i in t_type:
    if (i<3):
        original_plabel.append(0)
    else:
        original_plabel.append(1)


# In[ ]:
len(original_plabel)

# In[ ]:

df = pd.DataFrame({'x1':tsne_output[:,0],
                                       'x2':tsne_output[:,1],
                                       'plabels': tsne_plabel,
                                       'Process_model':original_plabel,
                                       'theta_type':t_type,

                                       })
# In[ ]:
df.shape
# In[ ]:
fig, (ax1) = plt.subplots(1,1,figsize=(36,16))
name="Step: 4 - Theta type"
#name="Step: 4 - ProcessModel type"
sns.scatterplot(
            x="x1", y="x2",
            hue="theta_type",
            palette=sns.color_palette("bright",8),
            data=df,
            legend="full",
            alpha=1,
            ax=ax1,
)
ax1.legend(bbox_to_anchor=(1, 0.5),fontsize='xx-large')

ax1.set_title(name,fontsize=50)
plt.savefig(name+'.png',dpi=300)
plt.show()

# In[ ]:



# In[ ]:
