import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Box
from sklearn.preprocessing import StandardScaler
from environment.state_scaler import extract_state_vector
from preprocessing.simulation import (
    simulate_volume_driven_market_all_agents,
    rescale_action,
    get_vendor_logit_model
)

class MarketEnv(Env):
    def __init__(self, shop, df, vendors, config):
        self.shop = shop
        self.df = df[df['SHOP'] == shop].copy().reset_index(drop=True)
        self.vendors = vendors
        self.config = config
        self.current_step = 0
        self.episode_length = config.get("EPISODE_LENGTH", 30)
        self.steps_per_episode = config.get("STEPS_PER_EPISODE", 30)

        self.dumping_mode = config.get("DUMPING_STRATEGY", False)
        self.dump_vendor = config.get("DUMP_VENDOR", "hc")
        self.dump_shop = config.get("DUMP_SHOP", "ghi")
        self.dump_entry_date = pd.to_datetime(config.get("DUMP_ENTRY_DATE", "2099-01-01"))

        self.scaler = self._build_scaler()
        self.price_bounds = self._get_price_bounds()

        stats = self.df.groupby('SHOP_VENDOR_NAME').agg(
            total_market_share=('Market Share (%)', 'sum')
        ).reset_index()
        base_vendor = stats.loc[stats['total_market_share'].idxmax(), 'SHOP_VENDOR_NAME']
        self.beta_0, self.beta_1, *_ = get_vendor_logit_model(self.df, base_vendor)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _get_price_bounds(self):
        min_p = self.df['EFFECTIVE_PRICE'].min()
        max_p = self.df['EFFECTIVE_PRICE'].max()
        return {v: (min_p, max_p) for v in self.vendors}

    def _build_scaler(self):
        raw_states = self.df[self.df['SHOP_VENDOR_NAME'].isin(self.vendors)]
        state_vectors = raw_states.apply(lambda r: extract_state_vector(r, self.df), axis=1).tolist()
        scaler = StandardScaler()
        scaler.fit(state_vectors)
        return scaler

    def _get_obs(self):
        step_df = self.df[self.df['OBSERVED_TIME'] == self.dates[self.current_step]]
        obs = {}
        for v in self.vendors:
            if self.dumping_mode and self.shop == self.dump_shop and v == self.dump_vendor:
                if self.dates[self.current_step] < self.dump_entry_date:
                    obs[v] = np.zeros(self.observation_space.shape) # placeholder obs
                    continue

                row = step_df[step_df['SHOP_VENDOR_NAME'] == v]
                if row.empty:
                    recent_row = self.df[self.df['SHOP_VENDOR_NAME'] == v].iloc[-1]
                else:
                    recent_row = row.iloc[0]
                obs[v] = self.scaler.transform([extract_state_vector(recent_row, self.df)])[0]
        return obs

    def reset(self):
        self.current_step = 0
        #self.dates = sorted(self.df['OBSERVED_TIME'].unique())
        return self._get_obs()

    def step(self, actions, predicted_shares=None, alpha=1.0):
        date = self.dates[self.current_step]
        market_slice = self.df[self.df['OBSERVED_TIME'] == date].copy()

        for v in self.vendors:
            if self.dumping_mode and self.shop == self.dump_shop and v == self.dump_vendor:
                if date < self.dump_entry_date:
                    continue # skip applying action before dump starts

            a = actions[v]
            a_rescaled = rescale_action(a, *self.price_bounds[v])
            market_slice.loc[market_slice['SHOP_VENDOR_NAME'] == v, 'EFFECTIVE_PRICE'] = a_rescaled

        market_sim = simulate_volume_driven_market_all_agents(
            market_slice, beta_0=self.beta_0, beta_1=self.beta_1
        )

        rewards = {}
        info = {}
        for v in self.vendors:
            row = market_sim[market_sim['SHOP_VENDOR_NAME'] == v]
            if not row.empty:
                sim_share = float(row['MarketShareSim'].values[0])
                price = float(row['EFFECTIVE_PRICE'].values[0])
            else:
                sim_row = self.df[self.df['SHOP_VENDOR_NAME'] == v]
                sim_share = float(sim_row['Market Share (%)'].iloc[-1]) / 100.0 if not sim_row.empty else 0.0
                price = float(sim_row['EFFECTIVE_PRICE'].iloc[-1]) if not sim_row.empty else 100.0

            pred_share = predicted_shares.get(v, 0.0) if predicted_shares else 0.0
            revenue_est = price * pred_share
            reward = (1 - alpha) * revenue_est + alpha * (-abs(pred_share - sim_share))

            rewards[v] = reward
            info[v] = {
                'price': price,
                'pred_share': pred_share,
                'sim_share': sim_share,
                'revenue_est': revenue_est,
                'reward': reward
            }

        self.current_step += 1
        done = self.current_step >= self.steps_per_episode
        next_obs = self._get_obs() if not done else None

        return next_obs, rewards, done, info
