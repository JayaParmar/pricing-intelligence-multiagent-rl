import numpy as np

def calculate_market_share(prices, alpha=-0.5):
    utilities = [alpha * p for p in prices]
    exp_utilities = np.exp(utilities - np.max(utilities))  # stability
    shares = exp_utilities / np.sum(exp_utilities)
    return shares

def calculate_profit(price, share, cost, total_demand):
    return (price - cost) * share * total_demand

def calculate_revenue(price, share, total_demand):
    return price * share * total_demand

def calculate_weighted_reward(profit, share, profit_weight=0.5):
    return profit_weight * profit + (1 - profit_weight) * share

# Placeholder for shared critic utilities (future use)
def get_shared_critic_observation(state_dict):
    """
    Construct a joint observation from multiple vendors' states.
    This is useful for centralized critics in MARL.
    """
    return np.concatenate([v for v in state_dict.values()], axis=0)
