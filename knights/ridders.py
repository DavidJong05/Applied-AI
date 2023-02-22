import csv
import random
import copy
import matplotlib.pyplot as plt

def mutate(table_setting: list) -> list:
    # swap two people at the table
    idx = range(len(table_setting))
    i1, i2 = random.sample(idx, 2)
    table_setting[i1], table_setting[i2] = table_setting[i2], table_setting[i1]
    return table_setting

def crossover(elite_parent: list, random_parent: list) -> list:
    # cross an elite parent with the rest of a random parent
    half = int(len(elite_parent)/2)
    child = elite_parent[:half] # half of the elite parent
    for knight in random_parent:
        if knight not in child:
            child.append(knight) # rest from random parent
    return child

def fitness(table_setting: list, knights: dict) -> float:
    # zeg tussen ridder A en ridder B, heeft een waarde die gelijk is aan (affiniteit van A naar B) * (affiniteit van B naar A).
    # De fitness functie is de som van de waarden voor al deze "tussenplekken".
    paired_list = pair_up(table_setting)
    multiplier = 0
    for pair in paired_list:
        if pair[0] in knights.keys() and pair[1] in knights.keys():
            for value in knights.get(pair[0]).items():
                if value[0] == pair[1]:
                    aff_1 = value[1]
                    break # doesnt stay in the loop after found, saves computations
            for value in knights.get(pair[1]).items():
                if value[0] == pair[0]:
                    aff_2 = value[1]
                    break # doesnt stay in the loop after found, saves computations
        multiplier += aff_1 * aff_2
    return multiplier

def pair_up(names: list) -> list:
    # makes a paired list of the knights
    paired_list = []
    for i, name in enumerate(names):
        if i == len(names)-1: # last in list pairs with first in list
            paired_list.append([name,names[0]])
            return paired_list
        paired_list.append([name, names[i+1]])


knights_dict = {}
names = []

with open('RondeTafel.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
     for index, row in enumerate(spamreader):
        if index == 1:
            names = row[1:]
        if index > 1:
            knights_dict[row[0]] = dict([(names[index], float(aff)) for index, aff in enumerate(row[1:])])


epochs = 200
pop = 200
amount_of_elites = 20
amount_of_randoms = 100
amount_of_mutations = 10
amount_of_crossovers = 80
population = []
best_settings = []
random.seed(0)

# Initialize the first population
for i in range(0, pop):
    individual = copy.deepcopy(names)
    random.shuffle(individual)
    population.append(individual)


for i in range(0,epochs):
    ranking = []
    for individual in population:
        ranking.append((fitness(individual, knights_dict), individual)) 

    ranking.sort(key=lambda x:x[0], reverse=True)
    best_settings.append(ranking[0])
    population = [name[1] for name in ranking[:amount_of_elites]] #best ranks

    # add randoms
    for i in range(0, amount_of_randoms):
        population.append(random.choice(ranking[amount_of_elites:])[1])

    # add mutations
    for i in range(0, amount_of_mutations):
        population.append(mutate(random.choice(ranking)[1]))

    # add crossovers
    for i in range(0, amount_of_crossovers): 
        elite_parent = random.choice(ranking[:amount_of_elites])[1]
        random_parent = random.choice(ranking[amount_of_elites:])[1]
        population.append(crossover(elite_parent,random_parent))


plt.plot(range(0, epochs), [aff[0] for aff in best_settings])
plt.title("Screeplot")
plt.xlabel("Iteration")
plt.ylabel("Max affinity")
plt.savefig("screeplot.png")    
print("Plot done")
best_settings.sort(key=lambda x:x[0], reverse=True)
bestest = best_settings[0]
print(bestest)