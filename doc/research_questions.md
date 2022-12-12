## POMDP

### Belief State Updater

- **When to update the belief state?**

In a POMDP, the belief state is updated at each timestep based on the observations the agent receives and the actions it takes. This allows the agent to maintain an up-to-date estimate of the underlying state of the world and make more informed decisions about how to act in its environment.

The exact process for updating the belief state will depend on the specific POMDP and the algorithms being used to solve it. However, in general, the belief state is updated by combining the current belief state with the observation and action taken at the current timestep to calculate a new belief state. This may involve using techniques such as Bayesian inference or Markov chain Monte Carlo to combine the current belief state with the new information and produce an updated belief state.

Overall, the belief state in a POMDP is updated at each timestep to allow the agent to maintain an up-to-date estimate of the underlying state of the world and make more informed decisions about how to act in its environment.

- **Mathematically, what are the the steps for updating the belief state in a POMDP ?**

Suppose we have a POMDP with a set of states $\mathcal{S}$, a set of observations $\mathcal{O}$, and a set of actions $\mathcal{A}$. Let $b_t$ be the belief state at timestep $t$, $o_t$ be the observation received at timestep $t$, and $a_t$ be the action taken at timestep $t$. We can update the belief state $b_t$ at timestep $t$ as follows:

1. First, we can define a matrix $T$ where $T_{i,j}$ is the probability of transitioning from state $i$ to state $j$ when taking action $a_t$.
2. We can then define a vector $O_o$ where $O_{o,i}$ is the probability of receiving observation $o$ when in state $i$.
3. Using these two matrices, we can update the belief state $b_t$ as follows:

$$b_{t+1} = T \cdot b_t \cdot O_o$$

This equation updates the belief state $b_t$ by taking the current belief state, multiplying it by the transition matrix $T$ to account for the effect of the action $a_t$, and then multiplying it by the observation matrix $O_o$ to account for the effect of the observation $o_t$. This produces a new belief state $b_{t+1}$ that reflects the updated estimate of the underlying state of the world based on the action taken and the observation received at timestep $t$
