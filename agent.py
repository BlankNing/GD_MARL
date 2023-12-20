import numpy as np
import torch
import os
from algorithms.facmac import FACMAClearner as FACMAC
from algorithms.maddpg import MADDPG
from algorithms.iddpg import IDDPG
from algorithms.MADDPG_lstm_agent import MADDPG_lstm_actor
from algorithms.test_fac import FAC
from algorithms.FACMAC_SCH import FACMACSCH
from algorithms.FACLSTM import FACLSTM
from algorithms.MADDPGLSTM import MADDPG_lstm as MADDPGLSTM
from algorithms.MADDPG_Trans import MADDPGTrans as MADDPGTrans
from algorithms.MADDPG_Attention import MADDPG_Attention as MADDPGAttenion


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        policy_name=getattr(self.args,"algorithm","MADDPG")
        self.policy_class={
            "FACMAC":{"policy":FACMAC(args,agent_id),"sep_action":True,"time_series":False},
            "MADDPG":{"policy":MADDPG(args,agent_id),"sep_action":False,"time_series":False},
            "IDDPG":{"policy":IDDPG(args,agent_id),"sep_action":True,"time_series":False},
            "MADDPGLSTMactor":{"policy":MADDPG_lstm_actor(args,agent_id),"sep_action":False,"time_series":True},
            "FAC":{"policy":FAC(args,agent_id),"sep_action":True,"time_series":False},
            "FACMAC_SCH":{"policy":FACMACSCH(args,agent_id),"sep_action":True,"time_series":False},
            "MADDPGLSTM":{"policy":MADDPGLSTM(args,agent_id), "sep_action":False,"time_series":True},           
            "MADDPGTrans":{"policy":MADDPGTrans(args,agent_id), "sep_action":False,"time_series":True},
            # add maddpg_attention
            "MADDPGATTENTION":{"policy":MADDPGAttenion(args,agent_id), "sep_action":False,"time_series":True}
        }
        
        self.policy=self.policy_class[policy_name]["policy"]
        self.sep_action=self.policy_class[policy_name]["sep_action"]
        self.time_series=self.policy_class[policy_name]["time_series"]

        

    def select_action(self, o, noise_rate, epsilon):
        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        if np.random.uniform() < epsilon:
            # epsilon-greedy solution, my personal thought is that's not enough.
            u=torch.rand(self.args.action_shape[self.agent_id],device=device)
            u=2*self.args.high_action*u-self.args.high_action

        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
            
            
            if self.sep_action:
                u = self.policy.actor_network(inputs)["actions"].squeeze(0)

            else:
                u = self.policy.actor_network(inputs).squeeze(0)

            noise = noise_rate*self.args.high_action * torch.randn(*u.shape,device=device)
            u += noise
        
        return u

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

