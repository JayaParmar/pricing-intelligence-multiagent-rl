import torch as T
import torch.nn.functional as F
from agent.buffer import ReplayBuffer
from agent.networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent:
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99, n_actions=1,
                 max_size=1000000, layer1_size=256, layer2_size=256,
                 batch_size=256, reward_scale=2, chkpt_dir='tmp/sac',
                 device='cpu', shared_obs=False):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.scale = reward_scale
        self.device = device
        self.shared_obs = shared_obs

        self.total_steps = 0
        self.alpha_anneal_start = 0
        self.alpha_anneal_end = 1000  # Curriculum shaping duration

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                                  name='actor', max_action=1.0, chkpt_dir=chkpt_dir, device=self.device)

        critic_input_dim = input_dims[0]  # Already joint if shared_obs

        self.critic_1 = CriticNetwork(beta, [critic_input_dim], n_actions=n_actions,
                                      name='critic_1', chkpt_dir=chkpt_dir, device=self.device)
        self.critic_2 = CriticNetwork(beta, [critic_input_dim], n_actions=n_actions,
                                      name='critic_2',  chkpt_dir=chkpt_dir, device=self.device)
        self.value = ValueNetwork(beta, [critic_input_dim], name='value', chkpt_dir=chkpt_dir, device=self.device)
        self.target_value = ValueNetwork(beta, [critic_input_dim], name='target_value', chkpt_dir=chkpt_dir, device=self.device)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.actor.device)
        price_action, _, share_logit = self.actor.sample_normal(state, reparameterize=False)

        price = price_action.cpu().detach().numpy()[0]
        share = T.sigmoid(share_logit).cpu().detach().numpy()[0]  # soft prediction in [0,1]

        return price, share

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_params = dict(self.target_value.named_parameters())
        value_params = dict(self.value.named_parameters())

        for name in value_params:
            value_params[name] = tau * value_params[name].clone() + \
                                 (1 - tau) * target_params[name].clone()

        self.target_value.load_state_dict(value_params)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()

    def get_curriculum_alpha(self):
        t = min(self.total_steps, self.alpha_anneal_end)
        return 1.0 - t / self.alpha_anneal_end

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.total_steps += 1
        alpha = self.get_curriculum_alpha()

        state, action, reward, new_state, done, pred_share, sim_share = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float32).to(self.actor.device)
        new_state = T.tensor(new_state, dtype=T.float32).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float32).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        action = T.tensor(action, dtype=T.float32).to(self.actor.device)

        ### Value Loss
        actions_v, log_probs_v, _ = self.actor.sample_normal(state, reparameterize=False)
        critic_val = T.min(
            self.critic_1(state, actions_v),
            self.critic_2(state, actions_v)
        ).view(-1)
        value = self.value(state).view(-1)
        value_target = (critic_val - log_probs_v.view(-1)).detach()
        value_loss = 0.5 * F.mse_loss(value, value_target)

        self.value.optimizer.zero_grad()
        value_loss.backward()
        self.value.optimizer.step()

        ### Actor Loss
        actions_pi, log_probs_pi, _ = self.actor.sample_normal(state, reparameterize=True)
        critic_val_pi = T.min(
            self.critic_1(state, actions_pi),
            self.critic_2(state, actions_pi)
        ).view(-1)
        actor_loss = (log_probs_pi.view(-1) - critic_val_pi).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        ### Critic Loss
        target_value = self.target_value(new_state).view(-1)
        target_value[done] = 0.0
        target_q = reward + self.gamma * target_value
        target_q = target_q.view(self.batch_size, 1)

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)

        critic_1_loss = F.mse_loss(q1, target_q)
        critic_2_loss = F.mse_loss(q2, target_q)

        self.critic_1.optimizer.zero_grad()
        critic_1_loss.backward(retain_graph=True)
        self.critic_1.optimizer.step()

        self.critic_2.optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
