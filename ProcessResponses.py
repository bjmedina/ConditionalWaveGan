# Bryan Medina

import numpy
import pandas
import matplotlib.pyplot as plt

# Keep track of what response belongs to what network
GRGAN   = ["https://drive.google.com/open?id=15HAclRropf3OxZfogA2JZHvxhRNsMfK-",
"https://drive.google.com/open?id=1HkBcgOVYPa_gyeIupRV_AVYB7xzBGfNg",
"https://drive.google.com/open?id=1pvtWch6eAozcnmdANx_HpEbhqalj7H8l"]

LSTMGAN = ["https://drive.google.com/open?id=16WI4SQqbOeCJT6Ro5xAdCVl6OG_3BnAD",
"https://drive.google.com/open?id=1DBhllRQoKDCV4godjU09MTTgMq0WQ0NK",
"https://drive.google.com/open?id=1BhwvnRg-G4vkvd6YG-DmZNbrBvvancTN"] 

EQUAL   = "They both sound equally realistic."

questions = ["2000 Epochs", "100 Epochs", "4000 Epochs"]

colors   = {"GRU":"#ff9999", "LSTM":"#66b3ff", "EQUAL":"#99ff99"}
# Keep track of the responses per questions
responses = pandas.read_csv("responses.csv")

## We really don't care about the timestamp so we'll drop it
responses = responses.drop(['Timestamp'], axis=1)

## Convert links to actual answer
for col in responses.columns:
	responses.loc[responses[col] == GRGAN[int(col)], col]   = "GRU"
	responses.loc[responses[col] == LSTMGAN[int(col)], col] = "LSTM"
	responses.loc[responses[col] == EQUAL, col]             = "EQUAL"

print(responses)

## Get counts of results for each question
for col in responses.columns:
	responses.groupby(col).count()

for col in responses.columns:
	vals = responses[col].value_counts().values

	labels   = responses[col].value_counts().index
	explodes = [0.1 if x == max(vals) else 0 for x in vals]
	l_colors   = [ colors[label] for label in labels]

	plt.title(questions[int(col)])

	plt.pie(responses[col].value_counts(), 
		explode=explodes, 
		labels=responses[col].value_counts().index, 
		autopct='%1.1f%%',
		shadow=True,
		startangle=90,
		colors=l_colors)
	plt.savefig(questions[int(col)])
	plt.clf()