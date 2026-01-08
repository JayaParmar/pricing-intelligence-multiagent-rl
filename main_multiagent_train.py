import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from environment.market_env import MarketEnv
from agent.sac_agent import Agent as SACAgent
from preprocessing.simulation import rescale_action
from config import CONFIG

def train_shop(shop, full_df, config):
    print(f"\n🔁 Training for SHOP: {shop}")
    df_shop = full_df[full_df['SHOP'] == shop].copy()
    vendors = df_shop['SHOP_VENDOR_NAME'].unique().tolist()

    model_dir = config['MODEL_DIR']
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("model_sac_pricing", exist_ok=True)
    output_csv = os.path.join("model_sac_pricing", f"{shop}_predicted.csv")

    # Skip if already trained
    all_files_exist = all(
        os.path.exists(os.path.join(model_dir, f"{shop}_{v}_actor_sac")) and
        os.path.exists(os.path.join(model_dir, f"{shop}_{v}_critic1_sac")) and
        os.path.exists(os.path.join(model_dir, f"{shop}_{v}_critic2_sac")) and
        os.path.exists(os.path.join(model_dir, f"{shop}_{v}_value_sac")) and
        os.path.exists(os.path.join(model_dir, f"{shop}_{v}_target_value_sac"))
        for v in vendors
    )
    if all_files_exist and not config.get('FORCE_RETRAIN', False):
        print(f"⚠️  Models already exist for {shop}. Skipping training.")
        return

    env = MarketEnv(shop, df_shop, vendors, config)
    agents = {}

    for v in vendors:
        prefix = f"{shop}_{v}"
        agent = SACAgent(
            input_dims=(env.observation_space.shape[0],),
            n_actions=1,
            alpha=config.get('ALPHA', 0.0003),
            beta=config.get('BETA', 0.0003),
            tau=config.get('TAU', 0.005),
            chkpt_dir=model_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        agent.actor.checkpoint_file = os.path.join(model_dir, f"{prefix}_actor_sac")
        agent.critic_1.checkpoint_file = os.path.join(model_dir, f"{prefix}_critic1_sac")
        agent.critic_2.checkpoint_file = os.path.join(model_dir, f"{prefix}_critic2_sac")
        agent.value.checkpoint_file = os.path.join(model_dir, f"{prefix}_value_sac")
        agent.target_value.checkpoint_file = os.path.join(model_dir, f"{prefix}_target_value_sac")

        agents[v] = agent

    total_steps = 0
    anneal_steps = config.get('CURRICULUM_STEPS', 1000)
    episode_log = []

    for episode in tqdm(range(config['EPISODES']), desc=f"Training {shop}"):
        obs = env.reset()

        for step in range(env.episode_length):
            actions, pred_shares, denorm_prices = {}, {}, {}
            for v in vendors:
                price_norm, share = agents[v].choose_action(obs[v])
                denorm_price = rescale_action(price_norm, *env.price_bounds[v])
                actions[v] = price_norm
                denorm_prices[v] = denorm_price
                pred_shares[v] = share

            alpha = 1.0 - min(total_steps / anneal_steps, 1.0)
            next_obs, _, done, info = env.step(actions, pred_shares, alpha=alpha)

            for v in vendors:
                pred_share = pred_shares[v].item()
                sim_share = info[v]['sim_share']
                price = denorm_prices[v]
                reward = (1 - alpha) * (price * pred_share) + alpha * (-abs(pred_share - sim_share))

                agents[v].memory.store_transition(
                    state=obs[v],
                    action=np.array([actions[v]]),
                    reward=reward,
                    new_state=next_obs[v] if next_obs else obs[v],
                    done=done,
                    pred_share=pred_share,
                    sim_share=sim_share
                )
                agents[v].learn()

                episode_log.append({
                    'SHOP': shop,
                    'VENDOR': v,
                    'EPISODE': episode,
                    'STEP': step,
                    'DAY': step + 1,
                    'PRED_PRICE': price,
                    'PRED_SHARE': pred_share,
                    'SIM_SHARE': sim_share,
                    'REWARD': price * pred_share,  # Business revenue
                    'ALPHA': alpha
                })

            total_steps += 1
            obs = next_obs
            if done:
                break

    for agent in agents.values():
        agent.save_models()

    pd.DataFrame(episode_log).to_csv(output_csv, index=False)
    print(f"✅ Saved predicted results to {output_csv}\n")

def main():
    full_df = pd.read_csv(CONFIG['DATA_PATH'])
    shops = full_df['SHOP'].unique()

    for shop in shops:
        train_shop(shop, full_df, CONFIG)

if __name__ == "__main__":
    main()
