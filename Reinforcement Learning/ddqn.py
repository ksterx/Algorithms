import random
from collections import deque

import torch


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Transition:
    def __init__(self, *args):
        self.args = args

    def __getitem__(self, item):
        return self.args[item]

    def __len__(self):
        return len(self.args)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()

        self.n_actions = n_outputs

        self.fc1 = torch.nn.Linear(n_inputs, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, n_outputs)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                return self(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)


class DQN:
    def __init__(self, model, lr, gamma, memory_size, batch_size):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                out = self.model(state)
                return self.model(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.model.n_actions)]], dtype=torch.long)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = torch.nn.functional.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


def main():
    import gymnasium as gym
    import torch

    env = gym.make("CartPole-v1", render_mode="human")

    model = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    dqn = DQN(model, lr=0.001, gamma=0.99, memory_size=10000, batch_size=32)

    n_episodes = 1000

    for ep in range(n_episodes):
        state, info = env.reset()
        done = False
        score = 0

        while not done:
            action = dqn.select_action(torch.FloatTensor(state), epsilon=0.5)
            next_state, reward, done, _, _ = env.step(action.item())
            score += reward

            if done:
                next_state = None

            dqn.memory.add(
                Transition(
                    torch.FloatTensor(state),
                    action,
                    torch.FloatTensor([next_state]),
                    torch.FloatTensor([reward]),
                )
            )

            state = next_state

            dqn.learn()

        print(f"Episode: {ep}, Score: {score}")


if __name__ == "__main__":
    main()
