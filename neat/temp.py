f = open('winner_neat_all.pkl', 'rb')
# convert binary to dictionary
import pickle
winner = pickle.load(f)
print(winner)
