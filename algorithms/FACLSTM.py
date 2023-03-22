import torch
import os
from modules.agents.lstm_agent import Actor as LSTMActor
from modules.critics.facmac import FACMACCritic as Critic



class FACLSTM:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = LSTMActor(args, agent_id)
        self.critic_network = Critic(agent_id, args)

        # build up the target network
        self.actor_target_network = LSTMActor(args, agent_id)
        self.critic_target_network = Critic(agent_id, args)


        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())


        # create the optimizer



        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name + '/share_param=' + str(
            self.args.share_param)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + self.args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))

            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))

        if torch.cuda.is_available():
            self.cuda()

    # soft update
    def _soft_update_target_network(self):  # 对target_network的更新
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)


    # update the network
    def train(self, transitions, other_agents):
        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32, device=device)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        r_s = r[:, -1].squeeze()
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项 seq_version
        o_s, u_s, o_next_s = [], [], []
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            o_s.append(transitions['o_%d' % agent_id][:, -1, :].squeeze())
            u.append(transitions['u_%d' % agent_id])
            u_s.append(transitions['u_%d' % agent_id][:, -1, :].squeeze())
            o_next.append(transitions['o_next_%d' % agent_id])
            o_next_s.append(transitions['o_next_%d' % agent_id][:, -1, :].squeeze())



        # calculate the target Q value function

        with torch.no_grad():
            # 得到下一个状态对应的动作

            u_next=self.actor_target_network.forward(o_next[self.agent_id])
            q_next = self.critic_target_network.forward(o_next_s, u_next).detach()


            target_q = (r_s.unsqueeze(1) + self.args.gamma * q_next).detach()


        q = self.critic_network.forward(o_s, u_s[self.agent_id])

        critic_loss = (target_q - q).pow(2).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # the actor loss

        actions_taken=self.actor_network.forward(o[self.agent_id])
        q_agent = self.critic_network.forward(o_s, actions_taken)
        q_val_of_actions_taken=q_agent

        actor_loss = -q_val_of_actions_taken.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = model_path + '/share_param=' + str(self.args.share_param) + '/' + self.args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(), model_path + '/' + num + '_critic_params.pkl')

    def cuda(self, device="cuda:0"):
        self.actor_network.cuda(device=device)
        self.actor_target_network.cuda(device=device)
        self.critic_network.cuda(device=device)
        self.critic_target_network.cuda(device=device)



