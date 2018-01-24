#python pair_learning.py -gpu 1 -props ./properties/probs.en.pair > result
#python top_pair_learning.py -gpu 1 -props ./properties/probs.en.toppair > result.top

#python reinforceAll.py -props ./properties/probs.en.rl -gpu 1 > result.rl
#python reinforceExpect.py -props ./properties/probs.en.rl -gpu 1 > result.rl
python reinforce_hierachi.py -props ./properties/probs.en.rl -gpu 2 > result.rl

