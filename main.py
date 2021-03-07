import gym
from utils import plot
import numpy as np
from td3 import Agent




if __name__ == '__main__':

    env = gym.make('HumanoidStandup-v2')

    agente = Agent(alpha=0.003, beta=0.003, 
                input_dims=env.observation_space.shape,
                tau=0.01, env=env, batch_size=256, 
                layer1_size=600, layer2_size=400,
                n_actions=env.action_space.shape[0])
    episodi = 100

    filename = 'HumanoidStandUp_v2_' + str(episodi) + '.png'
    plot_directory = 'plots/' + filename

    best_reward = env.reward_range[0]
    history = []

    
    '''
    Questo è il Loop che itera tante volte quanti episodi sono stati specificati questa sequanza di operazioni:
    Azione
    Osservazione
    Salvataggio nel buffer
    Apprendimento
    Storage di reward e observation
    '''

    for i in range(episodi):
        observation = env.reset()
        
        episode_reward = 0


        done = False


        while not done:
            action = agente.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agente.remember(observation, action, reward, observation_, done)
            agente.learn()
            episode_reward += reward
            observation = observation_

        history.append(episode_reward)   
        average_reward = np.mean(history[-100:])


        if average_reward > best_reward:
            best_reward = average_reward
            agente.save_models()


        print('episodio ', i, 'score %.2f' % episode_reward,
                '100 games  %.3f' % average_reward)





    x = [i+1 for i in range(episodi)]
    plot(x, history, plot_directory)
