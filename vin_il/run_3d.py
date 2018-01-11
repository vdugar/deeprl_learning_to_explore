import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from dataset_3d import GridworldDataManip, GridworldDataManipRollout
from vin_3d import VIN
import pickle

# Automatic swith of GPU mode if available
use_GPU = torch.cuda.is_available()

# Parsing training parameters
parser = argparse.ArgumentParser()

parser.add_argument('--datapath',
                    type=str,
                    default='./data/30_new/',
                    help='Path to data files')
parser.add_argument('--imsize',
                    type=int,
                    default=12,
                    help='Size of image')
parser.add_argument('--lr',
                    type=float,
                    default=0.0001,
                    help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
parser.add_argument('--epochs',
                    type=int,
                    default=51,
                    help='Number of epochs to train')
parser.add_argument('--k',
                    type=int,
                    default=20,
                    help='Number of Value Iterations')
parser.add_argument('--ch_i',
                    type=int,
                    default=2,
                    help='Number of channels in input layer')
parser.add_argument('--ch_h',
                    type=int,
                    default=150,
                    help='Number of channels in first hidden layer')
parser.add_argument('--ch_q',
                    type=int,
                    default=30,
                    help='Number of channels in q layer (~actions) in VI-module')
parser.add_argument('--batch_size',
                    type=int,
                    default=256,
                    help='Batch size')  # TODO: Divisibility to DataLoader
parser.add_argument('--max_difficulty',
                    type=int,
                    default=7,
                    help='Maximum difficulty')
parser.add_argument('--num_actions',
                    type=int,
                    default=26,
                    help='Number of actions')
parser.add_argument('--decay_rate',
                    type=float,
                    default=0.95,
                    help='Learning rate decay')
parser.add_argument('--max_episode_length',
                    type=int,
                    default=30,
                    help='Maximum path length possible')

args = parser.parse_args()

# Instantiate a VIN model
net = VIN(args)

if use_GPU:
    net = net.cuda()

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.RMSprop(net.parameters(), lr=args.lr, eps=1e-6)
learning_rate = args.lr

# Define dataset
trainset = GridworldDataManip(args.datapath, imsize=args.imsize, max_difficulty=args.max_difficulty, train=True)
testset = GridworldDataManip(args.datapath, imsize=args.imsize, max_difficulty=args.max_difficulty, train=False)
print 'Data loaded'

# Create dataloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# Save directory
save_dir = 'save/'

# Files
f = open('epoch_loss_acc_train.csv', 'w')
f_test = open('epoch_acc_test.csv', 'w')

for epoch in range(args.epochs):
    # for param_group in optimizer.param_groups:
    #    param_group['lr'] = learning_rate

    # learning_rate *= args.decay_rate
    running_losses = []
    running_acc = []

    start_time = time.time()
    for i, data in enumerate(trainloader):  # Loop over batches of data
        # Get input batch
        X, S1, S2, S3, labels = data

        if X.size()[0] != args.batch_size:  # TODO: Bug with DataLoader
            continue  # Drop those data, if not enough for a batch

        if use_GPU:
            X = X.cuda()
            S1 = S1.cuda()
            S2 = S2.cuda()
            S3 = S3.cuda()
            labels = labels.cuda()

        X, S1, S2, S3, labels = Variable(X), Variable(S1), Variable(S2), Variable(S3), Variable(labels)

        net.zero_grad()

        outputs = net(X, S1, S2, S3, args)

        _, predicted = torch.max(outputs, 1)
        predicted = predicted.data

        correct = (predicted == labels.data).sum()
        total = labels.data.size()[0]
        acc = (float(correct)/total)*100

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_losses.append(loss.data[0])
        running_acc.append(acc)

    time_duration = time.time() - start_time

    # Print epoch logs
    print('[Epoch # {:3d} ({:.1f} s)] Loss: {:.4f} Acc: {:.3f}'.format(epoch + 1, time_duration, np.mean(running_losses), np.mean(running_acc)))
    f.write(str(epoch+1) + ',' + str(np.mean(running_losses)) + ',' + str(np.mean(running_acc)) + '\n')

    # Run it on test once
    correct = 0.
    total = 0
    for i, data in enumerate(testloader):
        # Get inputs
        X, S1, S2, S3, labels = data

        if X.size()[0] != args.batch_size:  # TODO: Bug with DataLoader
            continue  # Drop those data, if not enough for a batch

        # Send Tensors to GPU if available
        if use_GPU:
            X = X.cuda()
            S1 = S1.cuda()
            S2 = S2.cuda()
            S3 = S3.cuda()
            labels = labels.cuda()

        # Wrap to autograd.Variable
        X, S1, S2, S3 = Variable(X, volatile=True), Variable(S1, volatile=True), Variable(S2, volatile=True), Variable(S3, volatile=True)

        # Forward pass
        outputs = net(X, S1, S2, S3, args)

        # Select actions with max scores(logits)
        _, predictetd = torch.max(outputs, dim=1)

        # Unwrap autograd.Variable to Tensor
        predicted = predicted.data

        # Compute test accuracy
        correct += (predicted == labels).sum()
        total += labels.size()[0]  # args.batch_size*num_batches, TODO: Check if DataLoader drop rest examples less than batch_size

    print('[Epoch # {:3d} Test Acc: {:.3f}'.format(epoch + 1, 100*(correct/total)))
    f_test.write(str(epoch+1) + ',' + str(100*(correct/total)) + '\n')

    if epoch % 10 == 0 and epoch != 0:
        # Save the model
        print 'Saving model'
        torch.save({'epoch': epoch,
                    'state_dict': net.state_dict()}, save_dir+str(epoch)+'.tar')

f.close()
f_test.close()
print('\nFinished training. \n')

'''
print 'Starting independent testing'
correct = 0.
total = 0
for i, data in enumerate(testloader):
    # Get inputs
    X, S1, S2, S3, labels = data

    if X.size()[0] != args.batch_size:  # TODO: Bug with DataLoader
        continue  # Drop those data, if not enough for a batch

    # Send Tensors to GPU if available
    if use_GPU:
        X = X.cuda()
        S1 = S1.cuda()
        S2 = S2.cuda()
        S3 = S3.cuda()
        labels = labels.cuda()

    # Wrap to autograd.Variable
    X, S1, S2, S3 = Variable(X), Variable(S1), Variable(S2), Variable(S3)

    # Forward pass
    outputs = net(X, S1, S2, S3, args)

    # Select actions with max scores(logits)
    _, predicted = torch.max(outputs, dim=1)

    # Unwrap autograd.Variable to Tensor
    predicted = predicted.data

    # Compute test accuracy
    correct += (predicted == labels).sum()
    total += labels.size()[0]  # args.batch_size*num_batches, TODO: Check if DataLoader drop rest examples less than batch_size

print('Test Accuracy (with {:d} examples): {:.2f}%'.format(total, 100*(correct/total)))
print('\nFinished testing.\n')
'''

print 'Starting rollout testing'
# ROLLOUT TESTING
testRollout = GridworldDataManipRollout(args.datapath, args.imsize, args.max_difficulty, train=False)
numSuccesses = [0. for _ in range(args.max_difficulty)]
numFailures = [0. for _ in range(args.max_difficulty)]
trajDiff = [0. for _ in range(args.max_difficulty)]
total = [0 for _ in range(args.max_difficulty)]
for i, data in enumerate(testRollout):
    X, start, goal, path, env, difficulty = data
    total[difficulty] += 1

    pred_path = [start]

    S1 = torch.Tensor([start[0]])
    S2 = torch.Tensor([start[1]])
    S3 = torch.Tensor([start[2]])

    if use_GPU:
        X = X.cuda()

    X = Variable(X.unsqueeze(0))

    for j in range(args.max_episode_length):
        if use_GPU:
            S1 = S1.cuda()
            S2 = S2.cuda()
            S3 = S3.cuda()

        S1, S2, S3 = Variable(S1), Variable(S2), Variable(S3)

        outputs = net(X, S1, S2, S3, args)

        _, predicted = torch.max(outputs, dim=1)

        predicted = predicted.data.cpu().numpy()

        state_and_coords, reward, done, _ = env._step(predicted[0, 0])
        coords = state_and_coords[1]

        pred_path.append(np.copy(coords))

        if done and reward == 1:
            numSuccesses[difficulty] += 1
            length_of_optimal_path = len(path)-1
            trajDiff[difficulty] += j + 1 - length_of_optimal_path

            # Save the env, true path, predicted path
            with open(save_dir+'success_'+str(int(numSuccesses[difficulty]))+'_'+str(difficulty)+'.pkl', 'wb') as f:
                pickle.dump((np.copy(env.map_data), path, pred_path), f)

            break

        elif done and reward == -1:
            numFailures[difficulty] += 1

            # Save the env, true path, predicted path
            with open(save_dir+'failure_'+str(int(numFailures[difficulty]))+'_'+str(difficulty)+'.pkl', 'wb') as f:
                pickle.dump((np.copy(env.map_data), path, pred_path), f)

            break

        S1 = torch.Tensor([coords[0]])
        S2 = torch.Tensor([coords[1]])
        S3 = torch.Tensor([coords[2]])

f_diff = open('difficulty_success_trajdiff.txt', 'w')
percent_success = [100 * (numSuccesses[i]/total[i]) for i in range(args.max_difficulty)]
avg_trajDiff = [trajDiff[i]/numSuccesses[i] for i in range(args.max_difficulty)]
str_write = [str(i+1)+','+str(percent_success[i])+','+str(avg_trajDiff[i])+'\n' for i in range(args.max_difficulty)]
for s in str_write:
    f_diff.write(s)

f_diff.close()

print('Percentage of success: {:.2f}'.format(np.mean(percent_success)))
print('Average trajectory length diff: {:.3f}'.format(np.mean(avg_trajDiff)))
