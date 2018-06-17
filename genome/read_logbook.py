import pickle
import pprint

with open('./genome/log/two_stage_full_logbook.pickle', 'rb') as f:
    logbook = pickle.load(f)

pprint.pprint(logbook.__dict__)
print(logbook)