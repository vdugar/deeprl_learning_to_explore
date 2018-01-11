#!/usr/bin/env python
'''
Run GridWorld environment with VIN
'''
import argparse

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd

from vin_3d import VIN_3d
from vin_3d_ac import VIN_AC_3d
from preprocessors import GridPreprocessor

import init_paths
from envs.grid_nd.NDGrid import NDGrid
import numpy as np
import matplotlib.pyplot as plt
import Tkinter
import torch.nn.functional as F

import io
from PIL import Image


def main():
    parser = argparse.ArgumentParser()

    # VIN parameters
    parser.add_argument('--imsize', type=int, default=8,
                        help='Size of the Gridworld')

    parser.add_argument('--n_obstacles', type=int, default=4,
                        help='Number of obstacles')

    parser.add_argument('--max_width_obstacles', type=int, default=3,
                        help='Max width of obstacles')

    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')

    parser.add_argument('--k', type=int, default=20,
                        help='Number of value iterations in VIN')

    parser.add_argument('--ch_i', type=int, default=2,
                        help='Number of channels in input layer')

    parser.add_argument('--ch_h', type=int, default=150,
                        help='Number of channels in first hidden layer')

    parser.add_argument('--ch_q', type=int, default=30,
                        help='NUmber of channels in q layer (~actions) in VI module')

    # RL parameters
    # Evaluation
    parser.add_argument('--eval',
                        help='Evaluate a trained model', action='store_true')

    # Training
    parser.add_argument('--replay_mem_size', type=int, default=1000000,
                        help='Replay memory size')

    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')

    parser.add_argument('--target_update_freq', type=int, default=5000,
                        help='Target network update frequency')

    parser.add_argument('--num_burn_in', type=int, default=10000,
                        help='Burn in')

    parser.add_argument('--train_freq', type=int, default=4,
                        help='Train frequency')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')

    parser.add_argument('--num_iterations', type=int, default=1000000,
                        help='Number of iterations to train')

    parser.add_argument('--max_episode_length', type=int, default=50,
                        help='Maximum length of an episode')

    parser.add_argument('--num_actions', type=int, default=8,
                        help='Number of actions in the env')

    parser.add_argument('--log_interval', type=int, default=5,
                        help='Logging frequency')

    parser.add_argument('--num_episodes', type=int, default=500000,
                        help='Maximum number of episodes')

    parser.add_argument('--validation_interval', type=int, default=500,
                        help='Validation interval')

    parser.add_argument('--num_validation_episodes', type=int, default=20,
                        help='Number of validation episodes')

    parser.add_argument('--L2_regularization', type=float, default=0.0005,
                        help='L2 regularization parameter')

    parser.add_argument('--save_every', type=int, default=10000,
                        help='Save frequency')

    parser.add_argument('--eval_episode', type=int, default=10000,
                        help='Evaluation episode')

    parser.add_argument('--plot_freq', type=int, default=2000,
                        help='Start Goal Plot frequency')

    parser.add_argument('--alg_type', type=str, default='reinforce',
                        help='Algorithm type. "reinforce" or "ac" (actor-critic)')

    parser.add_argument('--plot', type=int, default=1,
                        help='Enable plotting with 1, or disable with 0')
    parser.add_argument('--plot_dur', type=int, default=3,
                        help='Start-to-Goal Plot Continous Plots Count')

    parser.add_argument('--partial', help='Evaluate a trained model', action='store_true')

    args = parser.parse_args()

    # Environment parameters
    state_space_dims = 3
    state_space_size = args.imsize
    map_args = {
        'type': 'empty',
        'num_obs': args.n_obstacles,
        'max_width': args.max_width_obstacles,
        'difficulty': 1,
        'reveal_factor': 0.8
    }

    env = NDGrid(state_space_dims, state_space_size, map_args, partial_obs=args.partial)

    args.num_actions = len(env.actions)
    # Preprocessor
    preprocessor = GridPreprocessor(args.imsize)

    # Network
    if args.alg_type == 'reinforce':
        net = VIN_3d(args)
    elif args.alg_type == 'ac':
        net = VIN_AC_3d(args)
    net = net.cuda()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=args.L2_regularization)

    optimizer.zero_grad()
    net.zero_grad()

    if not args.partial:
        formula = lambda x: (1 - (x/100.) - 0.05)
    else:
        formula = lambda x: (1 - (2*x/35.))

    # target distribution
    target_distribution = Variable((torch.ones(args.num_actions)/(args.num_actions*1.0)).cuda())
    # exploration parameter
    alpha = 0.1

    def curriculum_update():
        # Update the curriculum
        # First, update the reveal factor
        # and then the difficulty
        '''
        if map_args['reveal_factor'] < 0.3:
            map_args['difficulty'] = min(6, map_args['difficulty'] + 1)
            map_args['reveal_factor'] = 0.7
        else:
            map_args['reveal_factor'] -= 0.1

        '''
        map_args['difficulty'] = min(args.imsize - 1, map_args['difficulty'] + 1)

        print 'Difficulty is', map_args['difficulty'], 'and reveal factor is', map_args['reveal_factor']
        env.set_map_args(map_args)

    def evaluate_agent():
        avg_reward = 0.
        for episode in range(args.num_validation_episodes):
            cum_reward = 0
            state = env.reset()
            path = []
            grid, coord = preprocessor.process_state_for_network(state)
            vis_grid = np.array(np.copy(grid[0]+grid[1]), dtype=np.int32)
            LOC = 1000
            vis_grid[int(coord[0]), int(coord[1])] = LOC
            path.append([coord[0], coord[1]])
            grid, coord = torch.Tensor(grid).cuda(), torch.Tensor(coord).cuda()
            for t in range(args.max_episode_length):
                if args.eval:
                    print coord
                action = select_action(grid, coord, infer=True)
                state, reward, done, _ = env.step(action[0, 0])
                grid, coord = preprocessor.process_state_for_network(state)
                LOC += 1
                vis_grid[int(coord[0]), int(coord[1])] = LOC
                path.append([coord[0], coord[1]])
                grid, coord = torch.Tensor(grid).cuda(), torch.Tensor(coord).cuda()
                cum_reward += reward
                if done:
                    break
            avg_reward += cum_reward
            # if episode % 4 == 0:
            #    plotenv(state, path)
            # if args.eval:
            # print vis_grid
            # fig = plt.figure()
            # ax1 = fig.add_subplot(121)
            # ax1.imshow(grid.cpu().numpy()[0], interpolation='nearest')
            # ax2 = fig.add_subplot(122)
            # ax2.imshow(grid.cpu().numpy()[1], interpolation='nearest')
            # plt.show()
        avg_reward /= args.num_validation_episodes
        if avg_reward > formula(map_args['difficulty']):
            curriculum_update()
        print 'Average Reward: {:.2f}'.format(avg_reward)

    def select_action_ac(grid, coord, plot=False, print_info=False, infer=False):
        ''' Actor-Critic'''
        probs, state_value = net(Variable(torch.unsqueeze(grid, 0)),
                                 Variable(torch.unsqueeze(coord, 0)), plot=plot)

        log_probs = torch.log(probs)
        kl_loss = alpha*net.kl_loss(log_probs, target_distribution)
        kl_loss.backward(retain_variables=True)

        if print_info:
            print probs
        action = probs.multinomial()
        if infer:
            _, action = torch.max(probs, 1)
        else:
            action = probs.multinomial()
            net.saved_actions.append((action, state_value))

        return action.data

    def select_action_re(grid, coord, plot=False, print_info=False, infer=False):
        '''REINFORCE'''
        probs = net(Variable(torch.unsqueeze(grid, 0)),
                    Variable(torch.unsqueeze(coord, 0)), plot=plot)

        # First compute KLDIVLOSS and backprop before doing anything
        log_probs = torch.log(probs)
        kl_loss = alpha*net.kl_loss(log_probs, target_distribution)
        kl_loss.backward(retain_variables=True)

        if print_info:
            print probs
        if infer:
            _, action = torch.max(probs, 1)
        else:
            action = probs.multinomial()
            net.saved_actions.append(action)
        return action.data

    def finish_episode_ac():
        '''Actor-Critic'''
        R = 0
        rewards = []
        saved_actions = net.saved_actions
        value_loss = 0
        for r in net.rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards).cuda()

        for (action, value),  r in zip(saved_actions, rewards):
            reward = r - value.data[0, 0]
            action.reinforce(reward)
            value_loss += F.smooth_l1_loss(value, Variable(torch.Tensor([r]).cuda()))

        optimizer.zero_grad()
        final_nodes = [value_loss] + list(map(lambda p: p[0], saved_actions))
        gradients = [torch.ones(1).cuda()] + [None] * len(saved_actions)

        autograd.backward(final_nodes, gradients)
        min_grad = np.Inf
        max_grad = -np.Inf
        torch.nn.utils.clip_grad_norm(net.parameters(), 10)
        for param in net.parameters():
            if torch.min(param.grad.data) < min_grad:
                min_grad = torch.min(param.grad.data)
            if torch.max(param.grad.data) > max_grad:
                max_grad = torch.max(param.grad.data)

        optimizer.step()
        del net.rewards[:]
        del net.saved_actions[:]
        return min_grad, max_grad

    def finish_episode_re():
        '''REINFORCE'''
        R = 0
        rewards = []
        for r in net.rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards).cuda()
        for action, r in zip(net.saved_actions, rewards):
            action.reinforce(r)
        optimizer.zero_grad()
        autograd.backward(net.saved_actions, [None for _ in net.saved_actions])
        min_grad = np.Inf
        max_grad = -np.Inf
        torch.nn.utils.clip_grad_norm(net.parameters(), 1)
        for param in net.parameters():
            if torch.min(param.grad.data) < min_grad:
                min_grad = torch.min(param.grad.data)
            if torch.max(param.grad.data) > max_grad:
                max_grad = torch.max(param.grad.data)
        optimizer.step()
        net.zero_grad()
        optimizer.zero_grad()
        del net.rewards[:]
        del net.saved_actions[:]
        return min_grad, max_grad

    # Float Env
    def plotenv(state, path):
        root = Tkinter.Tk()
        sz = state[0][1].shape[1]
        width = 800
        height = 800
        x = np.arange(0, width, width/sz)
        y = np.arange(0, height, height/sz)
        w = Tkinter.Canvas(root, width=width, height=height)
        w.pack()
        for j in range(sz):
            for i in range(sz):
                    if (state[0][1][j][i] == 10):
                        w.create_rectangle(y[i], x[j], width/sz+y[i], height/sz+x[j], fill="blue")
                        w.create_text(50+y[i], 50+x[j], fill="white", font="Times 14 bold", text="Goal")

                    elif ((i == int(path[0][1])) & (j == int(path[0][0]))):  # val[i][j].astype(int)==1:
                        w.create_rectangle(y[i], x[j], width/sz+y[i], height/sz+x[j], fill="orange")
                        w.create_text(50+y[i], 50+x[j], fill="darkblue", font="Times 14 bold", text="Start")

                    elif state[0][0][j][i] == 0:  # val[i][j].astype(int)==1:
                        w.create_rectangle(y[i], x[j], width/sz+y[i], height/sz+x[j], fill="green")

                    elif (state[0][0][j][i] == 1):
                        w.create_rectangle(y[i], x[j], width/sz+y[i], height/sz+x[j], fill="red")

        # print path
        for k in range(len(path)):
            if k < len(path)-1:
                w.create_line(50+100*int(path[k][1]), 50+100*int(path[k][0]), 50+100*int(path[k+1][1]), 50+100*int(path[k+1][0]))
                w.create_text(50+100*(int(path[k][1])+int(path[k+1][1]))/2, 50+100*(int(path[k][0])+int(path[k+1][0]))/2, fill="white", font="Times 14 bold", text=str(k+1))

        # @puneet logic to save image start-to-goal and closing the window automatically
        w.update()
        ps = w.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))

        # Save Image with episode name difficulty
        img.save('save/'+str(episode)+'_diff_'+str(map_args['difficulty'])+'.jpg')
        root.after(1000, lambda: root.destroy())
        root.after(500, root.mainloop())
        # root.mainloop()

    # set appropriate functions depending on algorithm type
    select_action = select_action_re
    finish_episode = finish_episode_re
    if args.alg_type == 'ac':
        # actor-critic stuff
        select_action = select_action_ac
        finish_episode = finish_episode_ac

    # Training
    if not args.eval:
        for episode in range(args.num_episodes):
            state = env.reset()
            grid, coord = preprocessor.process_state_for_network(state)
            grid, coord = torch.Tensor(grid).cuda(), torch.Tensor(coord).cuda()
            cum_reward = 0
            flag = True
            path = []
            executed_actions = []
            path.append([coord[0], coord[1]])
            for t in range(args.max_episode_length):
                if episode % args.plot_freq == 0 and episode != 0 and flag:
                    action = select_action(grid, coord, plot=(args.plot == 1), print_info=False)
                    flag = False
                else:
                    action = select_action(grid, coord, plot=False, print_info=False)
                executed_actions.append(env.actions[action[0, 0]])
                state, reward, done, _ = env.step(action[0, 0])
                grid, coord = preprocessor.process_state_for_network(state)
                grid, coord = torch.Tensor(grid).cuda(), torch.Tensor(coord).cuda()
                net.rewards.append(reward)
                cum_reward += reward

                # Append to path @puneet
                path.append([coord[0], coord[1]])

                if done:
                    break
            '''
            if args.plot == 1:
                if args.plot_dur > 2:
                    for d in range(args.plot_dur):
                        if episode % (args.plot_freq+d) == 0 and episode != 0:
                            # print "Path length=", len(path), " Transitions=", path
                            print
                            print "Executed actions=", executed_actions
                            #plotenv()
                # else:
                    # print args.plot_freq
                    # if episode % args.plot_freq == 0 and episode != 0:
                        # print "Path length=", len(path), " Transitions=", path
                        #plotenv()
            '''
            flag = True

            min_grad, max_grad = finish_episode()
            if episode % args.log_interval == 0:
                    print 'Episode {} \t Last length:{:5d} \t Cumulative Reward:{:.2f} \t Min grad:{:.4f} \t Max grad:{:.4f} \t Difficulty:{:2d} \t Value reqd:{:.2f} \t Reveal:{:.2f}'.format(episode, t, cum_reward, min_grad,
                                                                                                                                                                                               max_grad, map_args['difficulty'], formula(map_args['difficulty']), map_args['reveal_factor'])

            if episode % args.validation_interval == 0 and episode != 0:
                evaluate_agent()

            if episode % args.save_every == 0 and episode != 0:
                print 'Saving model'
                torch.save({
                    'iteration': episode,
                    'state_dict': net.state_dict()
                }, 'save/'+str(episode)+'.tar')

    else:
        print 'Loading model'
        ckpt = torch.load('save/'+str(args.eval_episode)+'.tar')
        net.load_state_dict(ckpt['state_dict'])
        evaluate_agent()

if __name__ == '__main__':
    main()
