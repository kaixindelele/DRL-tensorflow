try:
    from ddpg_sp.DDPG_per_class import DDPG as DDPG
    from sac_auto.sac_auto_per_class import SAC as SAC
    from td3_sp.TD3_per_class import TD3 as TD3    
except:
    from rl_algorithms.ddpg_sp.DDPG_per_class import DDPG as DDPG
    from rl_algorithms.sac_auto.sac_auto_per_class import SAC as SAC
    from rl_algorithms.td3_sp.TD3_per_class import TD3 as TD3
    
