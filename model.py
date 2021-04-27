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

