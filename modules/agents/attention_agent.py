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

class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.attention = AttentionModule(input_size=args.obs_shape[agent_id])
        self.fc1 = nn.Linear(args.obs_shape[agent_id], args.obs_shape[agent_id]) #?有必要搞个这个映射？
        self.fc2 = nn.Linear(args.obs_shape[agent_id], 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])


    def forward(self, x): #input shape[batch_size,seq_length,obs_shape] [1,16]
        attened_state=self.attention(x)
        x = F.relu(self.fc1(attened_state))
        x = F.relu(self.fc2(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        if len(actions.shape)==3:
            actions=actions[:,-1,:]
        return actions
