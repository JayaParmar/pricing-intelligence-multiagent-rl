import os

# Base and resolved paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'df_simulated_all.csv'))

CONFIG = {
    # Training
    'EPISODES': 100,
    'BATCH_SIZE': 1024,
    'USE_SHARED_CRITIC': True,       # Centralized critic toggle
    #'SEED': 42,
    'EPISODE_LENGTH': 30,
    'FORCE_RETRAIN': True,


    # Cost configuration
    'IVCLAR_COST': 35,
    'DEFAULT_COMPETITOR_COST': 10,
    'PROMO_THRESHOLD': 0.05,          # Price drop % considered a promotion

    # Paths
    'BASE_DIR': BASE_DIR,
    'DATA_PATH': DATA_PATH,
    'MODEL_DIR': os.path.join(BASE_DIR, 'models_sac_pricing'),

    # Dumping
    'DUMPING_STRATEGY': True,
    'DUMP_VENDOR': 'hc',
    'DUMP_SHOP': 'abc',
    'DUMP_ENTRY_DATE': '2024-11-26',
    'DUMP_END_DATE': '2025-01-31'
}
