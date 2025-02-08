# Learning to Steer Markovian Agents under Model Uncertainty

Code for the paper: [Learning to Steer Markovian Agents under Model Uncertainty](https://arxiv.org/abs/2407.10207).

## Instructions on Environment Configuration

This code is based on [PyTorch](https://pytorch.org/) and [stable_baseline3](https://stable-baselines3.readthedocs.io/en/master/). To setup the code, run the following commands:
```bash
git clone https://github.com/jiaweihhuang/Steering_Markovian_Agents.git

conda create --name Steer python=3.10.13
conda activate Steer

pip install numpy==1.26.4
pip install scipy==1.12.0
pip install torch==2.2.0 
pip install stable_baselines3==2.3.0
```

## Instructions on Running the Code

### Part 1: Experiments for known model setting

#### Experiments for normal-form Games
```python
# Stag Hunt
python run_NormalTraining.py --game StagHunt -K 2000000 --beta 25. --algo PPO -T 500 --lr 0.01 --obj MinGap --seed ...

python plot_trajectory.py --game StagHunt --obj MinGap -K 2000000 -T 500 --beta 25.0 --model-type Normal --grid-number 10

# Matching Pennies
python run_NormalTraining.py --game ZeroSum -K 2000000 --beta 25. --algo PPO -T 500 --lr 0.01 --obj Nash --seed ...

python plot_trajectory.py --game ZeroSum --obj MinGap -K 2000000 -T 500 --beta 25.0 --model-type Normal --grid-number 10
```

#### Experiments for the grid-world version of Stag Hunt

```python
# training
python StagHunt_GridWorld/run_game.py --train-batch-size 128 --steer-epochs-per-update 50 --lr 0.001 --beta 25 --seed ...

# evaluation
python StagHunt_GridWorld/evaluation.py --train-batch-size 128 --steer-epochs-per-update 50 --lr 0.001 --beta 25 --seed ...
```

### Part 2: Run the experiments for unknown model setting with small model set
#### Step 1: Training
```python
# oracle policy for f_{mu=0.7}
python run_Exp_SmallModel.py --game StagHunt -K 2000000 --beta 70.0 --algo PPO -T 500 --lr 0.01 --obj MinGap --mu 0.7 --sigma 0.3 --model-type Gaussian_lr --seed ...

# oracle policy for f_{mu=1.0}
python run_Exp_SmallModel.py --game StagHunt -K 2000000 --beta 20.0 --algo PPO -T 500 --lr 0.01 --obj MinGap --mu 1.0 --sigma 0.3 --model-type Gaussian_lr --seed ...

# belief state based policy
python run_Exp_SmallModel.py --game StagHunt -K 2000000 --beta 70.0 20.0 --algo PPO -T 500 --lr 0.01 --obj MinGap --mu 0.7 1.0 --sigma 0.3 --model-type Gaussian_lr --seed ...

```


### Part 3: Run the experiments for unknown model setting with large model set
#### Step 1: Train exploration policy
```python
# we treat 0.0 as +infty
python run_Strategic_Explore.py --game MP_Cooperative --num-players 10 -K 5000000 --beta 100.0 --algo PPO -T 30 --lr 0.01 --obj Explore --model-type Avaricious --sigma 0.5 --shift 0.0 -0.25 -0.75 --seed ...
```

#### Step 2: Evaluate random exploration strategy
```python
python run_Random_Explore.py -T 10 20 30 50 100 200 300 --num-eval 100 --game MP_Cooperative --num-players 10 --lr 0.01 --model-type Avaricious --shift 0.0 -0.25 -0.75
```

#### Step 3: Compare Oracle and FETE
```python
# 1. run the oracle policy
# --fixed-shift follows the threshold parameter for all agents, for example: --fixed-shift 0.0 0.0 0.0 0.0 0.0 -0.75 -0.75 -0.75 -0.75 -0.75
python run_Exp_LargeModel.py --game MP_Cooperative --num-players 10 -K 2000000 --beta 10.0 --algo PPO -T 500 --lr 0.01 --obj MaxUtility --sigma 0.5 --fixed-shift ... --seed ...

# 2.1 model estimation after random exploration
python model_estimation.py --game MP_Cooperative --num-players 10 -T 30 --lr 0.01 --sigma 0.5 --shift 0.0 -0.25 -0.75 --fixed-shift ... --seed ...

# 2.2 model estimation after FETE strategic exploration
python model_estimation.py --game MP_Cooperative --num-players 10 -T 30 --lr 0.01 --explore-policy-path Your_Exploration_Strategy --sigma 0.5 --shift 0.0 -0.25 -0.75 --fixed-shift ...  --seed ...

# 3. run the exploit policy
# here the --shift is the corresponding model identification results
python run_Exp_LargeModel.py --game MP_Cooperative --num-players 10 -K 2000000 --beta 10.0 --algo PPO -T 470 --lr 0.01 --obj MaxUtility --sigma 0.5 --fixed-shift ... --seed ...
```



## Citation

If you find the content of this repo useful, please consider citing:

```bibtex
@inproceedings{
    huang2025learning,
    title={Learning to Steer Markovian Agents under Model Uncertainty},
    author={Jiawei Huang and Vinzenz Thoma and Zebang Shen and Heinrich H. Nax and Niao He},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=IzYczpPqKq}
}
```


## Acknowledgements
Our code for the grid-world StagHunt experiments is based on the code from the following paper:
> Lu, C., Willi, T., De Witt, C. A. S., and Foerster, J. (2022). 
> Model-free opponent shaping. 
> In International Conference on Machine Learning, pages 14398–14411. PMLR.

Their original code can be found in <https://github.com/luchris429/model-free-opponent-shaping>.
