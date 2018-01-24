nohup python reinforce_hierachi.py -props ./properties/probs.en.rl -gpu 2 > result.rl.5 2>log.5 &
nohup python reinforce_hierachi.py -props ./properties/probs.en.rl.10 -gpu 3 -tb log_10 > result.rl.10 2>log.10 &
nohup python reinforceSingle.py -props ./properties/probs.en.rl -gpu 0 -tb log_s > result.rl 2>log.rl &

nohup python reinforce_hierachi.py -props ./properties/probs.en.rl -gpu 2 -reduced 1 > result.rl.5 2>log.5 &

