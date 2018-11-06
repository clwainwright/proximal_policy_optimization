# Proximal Policy Optimization

Proximal policy optimization is a reinforcement learning algorithm that works via a *policy gradient*. The original paper for the algorithm is [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) (see also the associated [OpenAI blog post](https://blog.openai.com/openai-baselines-ppo/)). This implementation is written for my own personal edification, but I hope that others find it helpful or informative.


# Installation and running

This project is based on Python 3, [Tensorflow](https://www.tensorflow.org), and the [OpenAI Gym environments](https://gym.openai.com). It's been tested on various Atari environments, although the basic algorithm can easily be applied to other scenarios.

To install the python requirements, run `pip3 install -r requirements.txt` (although you may want to create a [virtual environment](https://docs.python.org/3/tutorial/venv.html) first). The video recorder also requires [ffmepg](https://ffmpeg.org) which must be installed separately.

To run an environment, use e.g.

    python3 run_atari.py --logdir=./logdata --pfile=../example-pong-params.json

With the example parameters, the agent should be able to win a perfect game of Pong in about 2 million frames, which closely matches the results from the OpenAI baseline implementation. Other environments can be used by modifying the parameters file. To view the training progress, use tensorboard:

    tensorboard --logdir=./logdata


# Experimental modifications

One of the problems with policy gradient algorithms is that they are very sensitive to step size, and they are prone to catastrophic performance drops. Indeed, much of the original motivation for the PPO algorithm was to make the policy update robust across a larger range of step sizes. However, in my experiments with only modestly large step sizes I would frequently see performance drops from optimal to worse-than-random policies. This was easiest to reproduce in Pong, in which it's fairly straightforward to train an optimal agent. See *[coming soon!]* for a detailed analysis of one such drop and its proximal causes.

The implementation presented here includes two experimental modifications of the standard PPO algorithm which attempt to avoid catastrophic performance drops. Both are theoretically justifiable, but neither seem to eliminate the problem. However, I've so far only scratched the surface of their effects, and more thorough experimentation may prove them useful.

## Value function rescaling

The first modification rescales the gradient of the value loss function by a quantity that I'm calling the “pseudo-entropy,” `H’ = ∑π(a|s)(1-π(a|s))` where `π(a|s)` is the probability of taking a particular action in a particular state, and, like the standard entropy, the sum is over all possible actions. The pseudo-entropy is `1-1/N ≈ 1` when the distribution is uniform over `N` states and zero when the entropy is zero. The reason to do this rescaling is that the policy gradient contains a similar term when expanded to show the gradient with respect to the underlying logits. If the policy function is given by a softmax `π(a_i|s) = exp(x_i(s)) / ∑_j exp(x_j(s))`, then the policy update will look like

    ∇log π(a_i|s) = ∑_j (δ_ij - π(a_j|s)) ∇π(a_j|s)

The average magnitude of the term in parenthesis roughly corresponds to the pseudo-entropy. When the action is very certain, the update will on average be very small. This is necessary and expected: if the update weren't small, the probabilities would quickly saturate and the method would not converge to a good policy. However, a problem arises when we share weights between the value function and the policy function. In standard PPO there is no term that makes the value gradient correspondingly small, so if the policy is certain the weight updates will be driven by changes to the value function. Eventually, this may lead the agent away from a good policy. If the step size is moderately large, the agent may quickly cross over to a regime of bad policy and not recover.

This implementation of PPO includes three different types of value gradient rescaling:

- *per-state rescaling*: the pseudo-entropy is applied separately to each state, such that some states are effectively weighted much more heavily than others in the determination of the value update;
- *per-batch rescaling*: the average pseudo-entropy is calculated per mini-batch and applied uniformly across all updates in that batch;
- *smoothed rescaling*: the average pseudo-entropy is smoothed across multiple mini-batches and applied uniformly across all updates for each batch.

The per-state rescaling performs very poorly, and tends to result in agents that never learn good policies. The problem with per-state rescaling is that it prevents the agent from learning good value functions precisely in the states in which require the most critical (low entropy) actions.

The per-batch rescaling and smoothed rescaling perform similarly to each other. Unfortunately they don't appear to have a large effect on the catastrophic policy drops, and in general have little effect on the training. The problem here is that an optimal policy may have no preferences in certain states when all actions lead to similar rewards. Therefore, an optimal policy can have a high average entropy even though the actions are very certain in critical situations. Only in situations where the entropy is habitually very low does rescaling have a large effect, and those situations tend to have poor policies already.


## Modified Surrogate Objective

Proximal policy optimization builds upon standard policy gradient methods in two primary ways:

1. Rather than minimizing the standard loss `L_π(a, s) = -A log π(a|s)` where `A` is the observed advantage of the state-action pair, PPO introduces surrogate objective function `L'_π(a, s)`. The gradient of the surrogate function is designed to coincide with the original gradient when policy is unchanged from the prior time step. However, when the policy change is large, either the gradient gets clipped or a penalty is added such that further changes are discouraged.
2. The surrogate objective is minimized over several epochs over stochastic gradient descent for each batch of training data.

Combined, these two features yield good training with high sample efficiency, and, for the most part, without overly noisy policy updates and catastrophic policy drops.

The surrogate objective that's used in this implementation is the *clipped surrogate objective* (as opposed to the adaptive KL penalty which is also detailed in the original paper),

    L_{CLIP}(θ) = -E[ min(A r(θ), A clip(r(θ), 1-ε, 1+ε)) ]

where `r(θ) = π(a|s,θ) / π(a|s,θ_old)`. The choice of sign just denotes that I'm doing gradient *descent* rather than gradient *ascent*. Effectively, all the clipped function does is to produce a constant gradient until the policy has improved by a factor of `1+ε`, at which point the gradient goes to zero and further improvement stops.

There are a couple of things about this function that struck me a theoretically problematic. First, it's not symmetric. If an action is favored (positive advantage) and highly likely such that `π/π_old > 1-ε`, then it won't be clipped at all and the policy can increase arbitrarily close to one. If, on the other hand, an action has high probability but negative advantage, the surrogate won't clip until `π ≈ 1-ε`, which may represent a many order-of-magnitude increase in the policy's entropy. Either way, the clipping allows for very large changes in the underlying weights when `π_old` is close to one.

The second problem is that once an update moves a policy into the clipped regime, there is no counteracting force to bring it back towards the trusted region. This is especially problematic given that the weights are shared both across different states in the batch and with the value function, so the policy for a single state could be dragged far away from its old trusted value due to changes elsewhere in the network.

I have implemented two small changes to the clipped surrogate objective function which attempt to fix these problems and hopefully prevent catastrophic policy drops. The first change is to perform the clipping in logit space rather than probability space. We can rewrite the clipped loss as

    L_{CLIP}(θ) = E[ max(0, A (π' - π) / π_old) ] + const

where `π' = π_old (1 + ε sign(A))` is the target policy. Once the new policy moves beyond the target policy the function will be clipped and the gradient will be zero. To perform the clipping in logit space, we just need to move the target policy such that it's a fixed distance from the original policy in logit space:

    π' = π_old (1 + ε sign(A)) / (1 + ε sign(A) (2π_old - 1))

When `π_old = 1/2` the two formulations are equal, and when `π_old ≪ 1` the logit-space formulation has approximately twice change as the original. However, when `π_old ≈ 1` the new formulation provides a much tighter clipping. Note that the new formulation is symmetric: `π'(π_old, A) = 1 - π'(1-π_old, -A)`.

Initial experiments with this modified target did not show a large effect in the training (set the parameter `delta_target_policy = "logit"` to enable it). It didn't prevent a catastrophic policy drop, but it also didn't perform any worse than the original. I hope to do some more experiments to find out where, if anywhere, this might make a difference.

The second change aimed to draw the policy back towards the target policy if moves out of the trusted region. To do this, I replaced the clipped loss with

    L_{ABS}(θ) = E[ abs(A (π' - π) / π_old) ]

This, unfortunately, performed very poorly (set the parameter `peaked_policy_loss = True` to enable it). It led to very large variance in the policy gradient updates, and generally hastened rather than ameliorated the policy collapse.
