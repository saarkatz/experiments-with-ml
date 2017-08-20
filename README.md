# experiments-with-ml
Implementing and testing different approaches of machine learning

## Attempt number 1
Teach a neural network (NN) to play Connect Four using regression.
In the first attemp we build a simple (and wanky) api for creating neural networks using numpy matrices.
All activation functions were sigmoid (`scipy.special.expit`) and every layer had a bias node.
We then implemented backward propagation (BP) and a cost function ([Cross entrophy](https://en.wikipedia.org/wiki/Cross_entropy)) for the regression.
The idea was to let the NN play against itself and do small regression step (Using `scipy.optimize.fmin_cg`) for all the actions that resulted in a victory but this approach led to a network which plays the same move every time.

Things we tried in attempt to solve the problem (Nothing helped to solve it):
1. We added regularization temrs to the cost function and BP. Had an effect on the learning rate of the network but not particularly promising.
2. If the NN gets stuck and plays the same move, then it will always win only by doing this move (even if by chance) and so it will always learn only to play that specific move. For this reason we tried "unlearning" moves that were bad by doing a step in the positive direction of the gradient. This resulted in the NN learing to return an array of 1's. Not very useful.
3. We added the option for the NN to choose the action randomly based on the output. We used a function similar to [Boltzmann distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution) (Arbitrary choise) for the calculation of probability from the output.
4. Considering that maybe doing a single iteration of `fmin_cg` (by setting `maxiter=1`) is too large a step, we changed the lean function to simply subtruct the gradient calculated using BP, multiplied by `learn_rate` (Set to `1e-3` by default).
5. When unlearning the gradient tends to get larger as a move is unlearned more. The result is that the NN quickly gets to a state that of returning onlt 1's. For this reason we added a factor to decrease (the absolute value of) the reward for bad actions so that the rate at which actions are learned and unlearned is about the same. By virtue of trial and error we concluded that the magnitude for negetive reward should be about `1e-1` to `1e-2`. We also added a "running reward" using a discount factor, gamma, (As done in [Andrej Karpathy's blog post](http://karpathy.github.io/2016/05/31/rl/)) to decrease the reward as the action is farther removed from it. In the same context, in the case that the NN lost due to an illegal move we bumped up (I absolute terms) the negative reward and decreased gamma. This was because if a move was illegal then it makes less sense to penalize actions leading to it and it's an action we know the NN should'nt do no matter what strategy it uses. Later on it seemed that the NN actually tried to avoid going off the board before eventualy playing the same move.
6. Looking at the out put of the NN it seemd to change very little with the change of the input. For this reason we added an option to remove the bias so that the output will depend more on the input.
7. For the same reason we changed the activation function in all the layers to [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) (As used in [Andrej Karpathy's blog post](http://karpathy.github.io/2016/05/31/rl/))

None of these quite solved the problem and eventualy the NN always fell back to only playing the same move.

### Conclusion
Making machine learning work is a lot harder then it looks.

### Thoughts for the future
We probably will enhance and normalize the API.
We also consider trying a different approach such as Deep Q-Learning.
