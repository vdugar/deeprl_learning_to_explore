import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


def attention(tensor, params):
    """Attention model for grid world
    """
    S1, S2, S3, args = params

    num_data = tensor.size()[0]

    # Slicing S1 positions
    slice_s1 = S1.expand(num_data, args.ch_q, 1, args.imsize, args.imsize)
    q_out = tensor.gather(2, slice_s1).squeeze(2)

    # Slicing S2 positions
    slice_s2 = S2.expand(num_data, args.ch_q, 1, args.imsize)
    q_out = q_out.gather(2, slice_s2).squeeze(2)

    # Slicing S3 positions
    slice_s3 = S3.expand(num_data, args.ch_q, 1)
    q_out = q_out.gather(2, slice_s3).squeeze(2)

    return q_out


class VIN_AC_3d(nn.Module):
    """Value Iteration Network architecture"""
    def __init__(self, args):
        super(VIN_AC_3d, self).__init__()
        # First hidden Conv layer
        self.conv_h = nn.Conv3d(in_channels=args.ch_i,
                                out_channels=args.ch_h,
                                kernel_size=3,
                                stride=1,
                                padding=(3 - 1)//2,  # SAME padding: (F - 1)/2
                                bias=True)
        # Conv layer to generate reward image
        self.conv_r = nn.Conv3d(in_channels=args.ch_h,
                                out_channels=1,
                                kernel_size=3,
                                stride=1,
                                padding=(3 - 1)//2,  # SAME padding: (F - 1)/2
                                bias=False)
        # q layers in VI module
        self.conv_q = nn.Conv3d(in_channels=2,  # stack [r, v] -> 2 channels
                                out_channels=args.ch_q,
                                kernel_size=3,
                                stride=1,
                                padding=(3 - 1)//2,  # SAME padding: (F - 1)/2
                                bias=False)

        # Final fully connected layer for actions
        self.fc1 = nn.Linear(in_features=args.ch_q,  # After attention model -> Q(s, .) for q layers
                             out_features=args.num_actions,
                             bias=False)

        # Final fully connected layer for values
        self.fc2 = nn.Linear(in_features=args.ch_q,  # After attention model -> Q(s, .) for q layers
                             out_features=1,
                             bias=False)

        self.out_action = nn.Softmax()

        # Record grid image, reward image and its value images for each VI iteration
        self.grid_image = None
        self.reward_image = None
        self.value_images = []

        # Store arguments
        self.args = args
        self.counter = 0

        # Policy gradient stuff
        self.saved_actions = []
        self.rewards = []

        # KL div loss
        self.kl_loss = torch.nn.KLDivLoss()

    def init_weights(self):
        self.conv_h.weight.data.normal_(0, 0.01)
        self.conv_h.bias.data.fill_(0)

        self.conv_r.weight.data.normal_(0, 0.01)

        self.conv_q.weight.data.normal_(0, 0.01)

        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0)

    def forward(self, X, C, record_images=False, plot=False):
        # Get reward image from observation image
        h = self.conv_h(X)
        r = self.conv_r(h)

        if record_images:  # TODO: Currently only support single input image
            # Save grid image in Numpy array
            self.grid_image = X.data[0].cpu().numpy()  # cpu() works both GPU/CPU mode
            # Save reward image in Numpy array
            self.reward_image = r.data[0].cpu().numpy()  # cpu() works both GPU/CPU mode

        # Initialize value map (zero everywhere)
        v = torch.zeros(r.size())
        # Wrap to autograd.Variable
        v = Variable(v.cuda())

        # K-iterations of Value Iteration module
        for _ in range(self.args.k):
            rv = torch.cat([r, v], 1)  # [batch_size, 2, imsize, imsize]
            q = self.conv_q(rv)
            v, _ = torch.max(q, 1)  # torch.max returns (values, indices)

            if record_images:
                # Save single value image in Numpy array for each VI step
                self.value_images.append(v.data[0].cpu().numpy())  # cpu() works both GPU/CPU mode

        if plot:
            fig = plt.figure()
            ax1 = fig.add_subplot(141)
            ax1.imshow(X.data[0].cpu().numpy()[0], interpolation='nearest')
            ax2 = fig.add_subplot(142)
            ax2.imshow(X.data[0].cpu().numpy()[1], interpolation='nearest')
            ax3 = fig.add_subplot(143)
            ax3.imshow(r.data[0].cpu().numpy()[0], interpolation='nearest')
            ax4 = fig.add_subplot(144)
            ax4.imshow(v.data[0].cpu().numpy()[0], interpolation='nearest')
            plt.show()
        # Do one last convolution
        rv = torch.cat([r, v], 1)  # [batch_size, 2, imsize, imsize]
        q = self.conv_q(rv)

        # Attention model
        q_out = attention(q, [C[:, 0].long(), C[:, 1].long(), C[:, 2].long(), self.args])

        # Final Fully Connected layer
        logits = self.fc1(q_out)
        action = self.out_action(logits)
        value = self.fc2(q_out)

        return action, value
