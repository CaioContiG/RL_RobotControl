import numpy as np
import gym
from functools import partial
import class_maze as cm
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16,8)
import seaborn as sns
sns.set()

class TDlearning(object):
    def __init__(self, parameters):

        self.parameters = parameters

        # metodo
        self.method = parameters['method']

        # numero de episodios
        self.episode = 0

        # cria o ambiente
        self.env = cm.Maze()

        # tamanho dos espacos de estados e acoes
        self.num_states = np.prod(np.array(self.env.num_states))
        self.num_actions = self.env.action_space.n

        # parametros de aprendizado
        self.gamma = parameters['gamma']
        self.eps = parameters['eps']
        self.alpha = parameters['alpha']

        # log file (name depends on the method)
        self.logfile = parameters['q-file']
        if self.method == 'SARSA':
            self.logfile = 'RL_RobotControl/' + 'sarsa_' + self.logfile
        elif self.method == 'Q-learning':
            self.logfile = 'RL_RobotControl/' + 'qlearning_' + self.logfile
        else: print("Não salvou...")

        # reseta a politica
        self.reset()

    ##########################################
    # reseta a funcao acao-valor
    def reset(self):
        
        # reseta o ambiente
        S = self.env.reset()
        
        # Q(s,a)
        #random_values = [i for i in range(self.num_actions)]
        self.Q = np.zeros((self.num_states, self.num_actions))
        #self.Q = np.array([np.random.choice(random_values) for _ in range(self.num_states * self.num_actions)]).reshape((self.num_states , self.num_actions))
        
        # carrega tabela pre-computada se for o caso
        if self.parameters['load_Q']:
            try:
                with open(self.logfile, 'rb') as f:
                    data = np.load(f)
                    self.Q = data['Q']
                    self.episode = data['episodes']
            except: None

    ##########################################
    # retorna a politica corrente
    def curr_policy(self, copy=False):
        if copy:
            return partial(self.TabularEpsilonGreedyPolicy, np.copy(self.Q))
        else:
            return partial(self.TabularEpsilonGreedyPolicy, self.Q)
        
    ########################################
    # salva tabela Q(s,a)
    def save(self):
        with open(self.logfile, 'wb') as f:
            np.savez(f, Q=self.Q, episodes=self.episode)

    ##########################################
    def __del__(self):
        self.env.close()

class TDlearning(TDlearning):
    ##########################################
    # escolha da açao (epsilon-soft)
    def TabularEpsilonGreedyPolicy(self, Q, state):

        # acao otima corrente
        #Aast = Q[state, :].argmax()
        same_prob = np.argwhere(Q[state, :] == np.amax(Q[state, :])).ravel()
        Aast = np.random.choice(same_prob)
        
       # winner = np.argwhere(listy == np.amax(listy))
        # numero total de acoes
        nactions = Q.shape[1]
    
        # probabilidades de escolher as acoes
        p1 = 1.0 - self.eps + self.eps/nactions
        p2 = self.eps/nactions
        prob = [p1 if a == Aast else p2 for a in range(nactions)]
        c = np.random.choice(nactions, p=np.array(prob))
        #print(c)
        #print(same_prob, Aast, c)
        return c
    
class TDlearning(TDlearning):
    ##########################################
    def sarsa(self, S, A):

        # passo de interacao com o ambiente
        [Sl, R, done, _] = self.env.step(A)
        
        # escolhe A' a partir de S'
        Al = self.policy(Sl)
        
        # update de Q(s,a)
        self.Q[S, A] = self.Q[S, A] + self.alpha*(R + self.gamma*self.Q[Sl, Al] - self.Q[S, A])
        
        return Sl, Al, R, done

class TDlearning(TDlearning):
    ##########################################
    def qlearning(self, S):
        
        # \pi(s)
        A = self.policy(S)

        # passo de interacao com o ambiente
        [Sl, R, done, _] = self.env.step(A)
        
        self.Q[S, A] = self.Q[S, A] + self.alpha*(R + self.gamma*self.Q[Sl, :].max() - self.Q[S, A])
        
        return Sl, R, done
    
class TDlearning(TDlearning):
    ##########################################
    # simula um episodio até o fim seguindo a politica corente
    def rollout(self, max_iter=500, render=False):
        
        # inicia o ambiente (começa aleatoriamente)
        S = self.env.reset()
        
        # \pi(s)
        A = self.policy(S)

        # lista de rewards
        rewards = []

        for _ in range(max_iter):
            
            if self.method == 'SARSA':
                Sl, Al, R, done = self.sarsa(S, A)
                # proximo estado e ação
                #print(A)
                S = Sl
                A = Al
                
            elif self.method == 'Q-learning':
                Sl, R, done = self.qlearning(S)
                # proximo estado
                S = Sl

            # Salva rewards
            rewards.append(R)

            # renderiza o ambiente            
            if render:
                plt.subplot(1, 2, 1)                
                plt.gca().clear()
                self.env.render(self.Q)

            # chegou a um estado terminal?
            if done: break

        return rewards
    
class TDlearning(TDlearning):
    ##########################################
    def runEpisode(self):

        # novo episodio
        self.episode += 1

        # pega a politica corrente (on-policy)
        self.policy = self.curr_policy()

        # gera um episodio seguindo a politica corrente
        # render=((self.episode-1)%100 == 0)
        # render=((self.episode-1)%1000 == 0)
        if self.parameters['plot']:
            rewards = self.rollout(render=((self.episode-1)%250 == 0))
        else:
            rewards = self.rollout()
        
        # salva a tabela Q
        if self.parameters['save_Q']:
            self.save()

        return np.sum(np.array(rewards))
    
if __name__ == '__main__':
    n_execucoes = 30
    media_final = []
    for i in range(n_execucoes):
        print("##### TESTE ", i," #####")

        # ARMANDO
        plt.ion()

        # parametros
        parameters = {'episodes'  : 5000,
                    'gamma'     : 0.99,
                    'eps'       : 1.0e-2,
                    'alpha'     : 0.5,
                    'method'    : 'SARSA', #'SARSA' ou 'Q-learning'
                    'save_Q'    : True,
                    'load_Q'    : False,
                    'q-file'    : 'qtable.npy',
                    'plot'      : False}

        # TD algorithm
        mc = TDlearning(parameters)

        # historico de recompensas
        rewards = []
        avg_rewards = []
        if parameters['plot']:    
            plt.figure(1)
            plt.gcf().tight_layout()
        
        while mc.episode <= parameters['episodes']:
            # roda um episodio
            total_reward = mc.runEpisode()
            
            # rewrds
            rewards.append(total_reward)
            # reward medio
            avg_rewards.append(np.mean(rewards[-50:]))
            desvio_padrao = np.std(avg_rewards[-500:])
            
            if parameters['plot']:
                # plot rewards
                plt.subplot(1, 2, 2)
                plt.gca().clear()
                plt.gca().set_box_aspect(.5)
                plt.title('Recompensa por episódios')
                plt.plot(avg_rewards, 'b', linewidth=2)
                plt.plot(rewards, 'r', alpha=0.3)
                plt.xlabel('Episódios')
                plt.ylabel('Recompensa')
                plt.show()
                plt.pause(0.01)

            if mc.episode%250 == 0 or mc.episode == 1:
                print("Episode: ", mc.episode, ", Avg rewards: ", avg_rewards[-1], 'DP: ', desvio_padrao)

            np.save('RL_RobotControl/reward_file', rewards)
            np.save('RL_RobotControl/avg_rewards_file', avg_rewards)
        media_final.append(avg_rewards[-1])
        params = str(parameters['method']) + "_" + str(parameters['gamma']) + "_" + str(parameters['eps']) + "_" + str(parameters['alpha'])
        np.save('RL_RobotControl/rfinal_'+params, media_final)
        print(media_final)
        print('Media das medias: ', np.average(media_final), 'DP: ', np.std(media_final))
        plt.ioff()