import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        """

        :param obs_space:
        :param action_space:
        :param use_memory:
        :param use_text:
        """
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]       # calculate the image embedding size
        m = obs_space["image"][1]       # based on the LxH of the observation space
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory             - NOT CURRENTLY USED
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding     - NOT CURRENTLY USED
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        """
        Calculate the forward pass through the network based on the provided inputs (and memory, if using LSTM).

        :param obs: observation input to the network
        :param memory: memory state of the network (from the previous iteration)
        :return: tuple (action distribution, value, memory)
                 action distribution - soft-max distribution over the actions
                 value - value of current state
                 memory - memory state of the network (for the current iteration)
        """
        x = obs.image.transpose(1, 3).transpose(2, 3)   # transpose the inputs for compatibility
        x = self.image_conv(x)                          # pass the processed image through the convolutional layers
        x = x.reshape(x.shape[0], -1)                   # flatten the image into a single dimension

        if self.use_memory:                             # if we are using the LSTM
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:                                           # otherwise, passthrogh
            embedding = x

        if self.use_text:                               # if we are using the text input
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)                       # process the actor to generate the probability distribution
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)                      # process the critic to generate the value function
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        """
        Process the text to create the text embedding (via the RNN).

        :param text: text to process
        :return: embedded text implementation
        """
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]



class MultiQModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, reward_size=1):
        """

        :param obs_space:
        :param action_space:
        :param use_memory:
        :param use_text:
        """
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.reward_size = reward_size

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]       # calculate the image embedding size
        m = obs_space["image"][1]       # based on the LxH of the observation space
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size

        # Define actor's model
        self.q = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n*self.reward_size)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        """
        Calculate the forward pass through the network based on the provided inputs (and memory, if using LSTM).

        :param obs: observation input to the network
        :param memory: memory state of the network (from the previous iteration)
        :return: tuple (action distribution, value, memory)
                 action distribution - soft-max distribution over the actions
                 value - value of current state
                 memory - memory state of the network (for the current iteration)
        """
        x = obs.image.transpose(1, 3).transpose(2, 3)   # transpose the inputs for compatibility
        x = self.image_conv(x)                          # pass the processed image through the convolutional layers
        x = x.reshape(x.shape[0], -1)                   # flatten the image into a single dimension

        if self.use_memory:                             # if we are using the LSTM
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:                                           # otherwise, passthrogh
            embedding = x

        x = self.q(embedding)                      # process the critic to generate the value function
        #value = x.squeeze(1)
        value = x.reshape(-1, x.shape[1] // self.reward_size, self.reward_size)

        return value, memory

    def _get_embed_text(self, text):
        """
        Process the text to create the text embedding (via the RNN).

        :param text: text to process
        :return: embedded text implementation
        """
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

class MOWeightModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False,
                 reward_size=1, n_weights=11):
        """

        :param obs_space:
        :param action_space:
        :param use_memory:
        :param use_text:
        """
        super().__init__()

        # Decide which components are enabled
        self.use_memory = use_memory
        self.reward_size = reward_size

#        self.weights = _generate_convex_weights(self.reward_size, n_weights)
        self.weights = torch.vstack([torch.linspace(0,1,n_weights),
                                    torch.linspace(1,0,n_weights)]).T

        self.current_weight = 0
        self.counter = 0

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]       # calculate the image embedding size
        m = obs_space["image"][1]       # based on the LxH of the observation space
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory             - NOT CURRENTLY USED
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size

        self.actors = nn.ModuleList(); self.critics = nn.ModuleList()

        for w in self.weights:
            # Define actor's model
            self.actors.append(nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, action_space.n)
            ))
            # Define critic's model
            self.critics.append(nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, self.reward_size)
            ))

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory, w=None, scalarize=True):
        """
        Calculate the forward pass through the network based on the provided inputs (and memory, if using LSTM).

        :param obs: observation input to the network
        :param memory: memory state of the network (from the previous iteration)
        :return: tuple (action distribution, value, memory)
                 action distribution - soft-max distribution over the actions
                 value - value of current state
                 memory - memory state of the network (for the current iteration)
        """

        if w is None:
            self.counter=(self.counter+1)%1000
            if self.counter==0:
                self.current_weight=(self.current_weight+1)%len(self.weights)
            w = self.weights[self.current_weight,:]
        elif w in self.weights:
            self.current_weight = torch.all(self.weights==w, dim=1).nonzero().item()
        else:
            dists = []
            values = []
            multi_values = []
            for _w in self.weights:
                d, v, m = self.forward(obs, memory, w=_w, scalarize=False)
                dists.append(d)
                multi_values.append(v)
                values.append(torch.dot(w,multi_values[-1].squeeze()))

            opt_act = torch.argmax(torch.tensor(values)).item()

            #### this is only temporary ####
            temp = torch.vstack(multi_values);
            is_pareto = is_pareto_efficient_dumb(temp);
            import matplotlib.pyplot as plt
            plt.figure(2).clf();
            plt.plot(temp[:, 0], temp[:, 1], 'r.');
            plt.plot(temp[is_pareto, 0], temp[is_pareto, 1], 'b*');
            plt.xlabel('performance reward')
            plt.ylabel('safety reward')
            plt.show()
            #### this is only temporary ####
            return dists[opt_act], values[opt_act], m # dummy for now

            #return dists, values, m



        x = obs.image.transpose(1, 3).transpose(2, 3)   # transpose the inputs for compatibility
        x = self.image_conv(x)                          # pass the processed image through the convolutional layers
        x = x.reshape(x.shape[0], -1)                   # flatten the image into a single dimension

        if self.use_memory:                             # if we are using the LSTM
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:                                           # otherwise, passthrogh
            embedding = x

        x_act = self.actors[self.current_weight](embedding)                       # process the actor to generate the probability distribution
        dist = Categorical(logits=F.log_softmax(x_act, dim=1))

        x_crit = self.critics[self.current_weight](embedding)                      # process the critic to generate the value function
#        value = x.squeeze(1)
        multi_value = x_crit.reshape(-1, self.reward_size)

        value = torch.matmul(multi_value,w)

        if scalarize:
            return dist, value, memory
        else:
            return dist, multi_value, memory


def _generate_convex_weights(v, n):
    W_temp = torch.rand(n, v, device='cpu') #'cuda:0')
    W_temp[:,-1] = 1
    W_temp = torch.sort(W_temp).values

    W = torch.vstack([W_temp[:, i] - W_temp[:, i - 1] if i > 0 else
                        W_temp[:, i]
                        for i in range(v)]).T

    return W.float()

def is_pareto_efficient_dumb(values):
    """
    Find the pareto-efficient points
    :param values: An (n_points, n_values) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = torch.ones(values.shape[0], dtype = bool)
    for i, v in enumerate(values):
        is_efficient[i] = torch.all(torch.any(values[:i]<v, axis=1)) and torch.all(torch.any(values[i+1:]<v, axis=1))
    return is_efficient