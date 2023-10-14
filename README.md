# RL_RobotControl
Reinforcement Learning for Robot Control

Executar TD_learning_maze:
    Por padrão, não irá plotar a evolução por episódios. Para ativar, alterar o parâmetro "plot" para True.

Executar read_npy.py:
    Lê os resultados salvos em:
        reward_file.npy
        avg_rewards_file.npy
        sarsa_qtable.npy    
    e plota o resultado final. Não depende do parâmetro "plot".

Obs.: Não é necessário esperar o fim da execução para realizar o plot, o programa salva todos os dados a cada episódio. Portanto, caso pare antes do esperado, o plot realizado será até o último episódio concluído.