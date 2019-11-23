import re
import string
import pandas
from autocorrect import Speller

g_spell = Speller(lang='en')

def correct_spelling(tokens):
	return [g_spell(word) for word in tokens.split()]

def remove_punctuation(text):
    return re.sub('['+string.punctuation+']', '', text)

pos_set = []
for line in open('./data/pos', 'r'):
    pos_set.append(line.strip())

neg_set = []
for line in open('./data/neg', 'r'):
	neg_set.append(line.strip())

training_set = pos_set[:int(len(pos_set))]
training_set.extend(neg_set[:int(len(neg_set))])

label = [1 if i < len(training_set) / 2 else 0 for i in range(len(training_set))]

df = pandas.DataFrame(data={'Text': training_set, 'Label': label})
df['Text'].apply(lambda x: remove_punctuation(x))
df['Text'].apply(lambda x: correct_spelling(x))
df.to_csv('./data/data.csv', sep=',', index=False)