import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, input_size):
        super(AttentionModule, self).__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_weights = F.softmax(torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(x.size(-1)).float()), dim=-1)
        output = torch.bmm(attention_weights, v)

        return output

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.attention = AttentionModule(input_size=sum(args.obs_shape) + sum(args.action_shape))
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), sum(args.obs_shape) + sum(args.action_shape))
        self.fc2 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        self.q_out = nn.Linear(64, 1)#output q score of certain action under certain observation

    def forward(self, state, action):
        state = torch.cat(state, dim=-1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=-1)
        x = torch.cat([state, action], dim=-1)
        # print('attention_critic(s,a,x)',state.shape,action.shape,x.shape) #[16,5,48] [16,5,15] [16,5,63]
        attended_input=self.attention(x)
        x = F.relu(self.fc1(attended_input))
        x = F.relu(self.fc2(x))
        # print('attention_critic(s,a,x)',state.shape,action.shape,x.shape) [16,5,48] [16,5,15] [16,5,64]

        q_value = self.q_out(x)[:,-1,:]
        # print('att_q',q_value.shape) [16,1]
        return q_value
