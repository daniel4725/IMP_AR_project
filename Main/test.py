import pickle
from table_handling import CornersFollower

with open("corner_follower_l.sav", 'rb') as handle:
    test = pickle.load(handle)
    print("hello")