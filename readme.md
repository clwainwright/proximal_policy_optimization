# Proximal Policy Optimization

Proximal policy optimization is a reinforcement learning algorithm that works via a *policy gradient*. The original paper for the algorithm is [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) (see also the associated [OpenAI blog post](https://blog.openai.com/openai-baselines-ppo/)). This implementation is written for my own personal edification, but I hope that others find it helpful or informative.


# Installation and running

This project is based on Python 3, [Tensorflow](https://www.tensorflow.org), and the [OpenAI Gym environments](https://gym.openai.com). It's been tested on various Atari environments, although the basic algorithm can easily be applied to other scenarios.

To install the python requirements, run `pip3 install -r requirements.txt` (although you may want to create a [virtual environment](https://docs.python.org/3/tutorial/venv.html) first). The video recorder also requires [ffmepg](https://ffmpeg.org) which must be installed separately.

To run an environment, use e.g.

    python3 run_atari.py --logdir=./logdata --pfile=../example-pong-params.json

With the example parameters, the agent should be able to win a perfect game of Pong in about 2 million frames, which closely matches the results from the OpenAI baseline implementation. Other environments can be used by modifying the parameters file. To view the training progress, use tensorboard:

    tensorboard --logdir=./logdata
