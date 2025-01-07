from agents import QLearningAgent, PolicyIterationAgent, ActorCriticAgent
from snake_game import Snake
import matplotlib.pyplot as plt
import numpy as np

def Q_Snake():
    # Initialize and train a Q-learning agent
    q_agent = QLearningAgent()
    iterations, q_table, changes_over_time, survival_times = q_agent.train()
    
    # game = Snake(visualize=True)      # Initialize the game with visuals
    # game.q_game(iterations, q_table)  # Load and run the game using the trained Q-table

    # Print the average score across 10 games for this trained Snake
    scores = [Snake(visualize=False).q_game(iterations, q_table) for _ in range(10)] 
    print(scores)
    print(sum(scores)/len(scores))

    return changes_over_time, survival_times

def Policy_Snake():
    # Initialize and train a Policy Iteration agent
    policy_iteration_agent = PolicyIterationAgent()
    iterations, policy = policy_iteration_agent.train()
    # Load and run the game using actions from the Policy table
    game = Snake(visualize=True)
    game.policy_game(iterations, policy)

    scores = [Snake(visualize=False).policy_game(iterations, policy) for _ in range(10)] 
    print(scores)
    print(sum(scores)/len(scores))

def AC_Snake():
    ac_agent = ActorCriticAgent()
    iterations, policy_table, changes_over_time, survival_times = ac_agent.train()

    # game = Snake(visualize=True)                  # Initialize the game with visuals
    # game.ac_game(str(iterations), policy_table)   # Load and run the game using the trained Q-table

    # Print the average score across 10 games for this trained Snake
    scores = [Snake(visualize=False).ac_game(str(iterations), policy_table) for _ in range(10)] 
    print(scores)
    print(sum(scores)/len(scores))

    return changes_over_time, survival_times

if __name__ == "__main__":
    # train the agents
    # NOTE: to see how the agents play Snake after training, go to their functions and uncomment the game playing code
    # Policy_Snake()
    q_scores, q_survival = Q_Snake()
    ac_scores, ac_survival = AC_Snake()

    # Evaluation of average score
    q_scores_avg = [np.mean(band) for band in np.array_split(q_scores, len(q_scores) // 100)]
    ac_scores_avg = [np.mean(band) for band in np.array_split(ac_scores, len(ac_scores) // 100)]

    plt.title('Average score (at end of 100 steps)')
    plt.plot(range(0, len(q_scores_avg)), q_scores_avg, label='Q-Learning')
    plt.plot(range(0, len(ac_scores_avg)), ac_scores_avg, label='Actor-Critic')
    plt.legend()
    plt.xlabel('Evaluation step')
    plt.ylabel('Average score')
    plt.show()

    # Evaluation of average survival time
    q_survival_avg = [np.mean(band) for band in np.array_split(q_survival, len(q_survival) // 100)]
    ac_survival_avg = [np.mean(band) for band in np.array_split(ac_survival, len(ac_survival) // 100)]

    plt.title('Average survival length (in step count)')
    plt.plot(range(0, len(q_survival_avg)), q_survival_avg, label='Q-Learning')
    plt.plot(range(0, len(ac_survival_avg)), ac_survival_avg, label='Actor-Critic')
    plt.legend()
    plt.xlabel('Evaluation step')
    plt.ylabel('Average score')
    plt.show()
