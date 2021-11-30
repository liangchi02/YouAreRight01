import numpy
import GA
import MLP
import pickle
import matplotlib.pyplot
import mnist

x_train = mnist.train_images()
y_train = mnist.train_labels()
x_test = mnist.test_images()
y_test = mnist.test_labels()

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

X_train = (x_train / 256)
y_train = y_train/1.0
X_test = (x_test / 256)
y_test = y_test/1.0

print('X_train shape', x_train.shape)
print('y_train shape', y_train.shape)
print('X_test shape', x_test.shape)
print('y_test shape', y_test.shape)


sol_per_pop = 8
num_parents_mating = 4
num_generations = 2000
mutation_percent = 15

# Creating the initial population.
initial_pop_weights = []
for curr_sol in numpy.arange(0, sol_per_pop):
    HL1_neurons = 150
    input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1,size=(X_train.shape[1], HL1_neurons))
    HL2_neurons = 100
    HL1_HL2_weights = numpy.random.uniform(low=-0.1, high=0.1,size=(HL1_neurons, HL2_neurons))
    output_neurons = 10
    HL2_output_weights = numpy.random.uniform(low=-0.1, high=0.1,size=(HL2_neurons, output_neurons))
    initial_pop_weights.append(numpy.array([input_HL1_weights,HL1_HL2_weights,HL2_output_weights]))
pop_weights_mat = numpy.array(initial_pop_weights)
pop_weights_vector = GA.mat_to_vector(pop_weights_mat)

best_outputs = []
accuracies = numpy.empty(shape=(num_generations))

for generation in range(num_generations):
    print("Generation : ", generation)
    # converting the solutions from being vectors to matrices.
    pop_weights_mat = GA.vector_to_mat(pop_weights_vector,
                                       pop_weights_mat)
    # Measuring the fitness of each chromosome in the population.
    fitness = MLP.fitness(pop_weights_mat, X_train,y_train,activation="sigmoid")
    accuracies[generation] = fitness[0]
    print("Fitness")
    print(fitness)
    # Selecting the best parents in the population for mating.
    parents = GA.select_mating_pool(pop_weights_vector,fitness.copy(),num_parents_mating)
    print("Parents")
    print(parents)
    # Generating next generation using crossover.
    offspring_crossover = GA.crossover(parents,offspring_size=(pop_weights_vector.shape[0] - parents.shape[0], pop_weights_vector.shape[1]))
    print("Crossover")
    print(offspring_crossover)
    # Adding some variations to the offsrping using mutation.
    offspring_mutation = GA.mutation(offspring_crossover,mutation_percent=mutation_percent)
    print("Mutation")
    print(offspring_mutation)
    # Creating the new population based on the parents and offspring.
    pop_weights_vector[0:parents.shape[0], :] = parents
    pop_weights_vector[parents.shape[0]:, :] = offspring_mutation
pop_weights_mat = GA.vector_to_mat(pop_weights_vector, pop_weights_mat)
best_weights = pop_weights_mat[0, :]
acc, predictions = MLP.predict_outputs(best_weights, X_train, y_train, activation="sigmoid")
print("Accuracy of the best solution is : ", acc)
print(accuracies)
matplotlib.pyplot.plot(accuracies, linewidth=5, color="black")
matplotlib.pyplot.xlabel("Iteration", fontsize=15)
matplotlib.pyplot.ylabel("Fitness", fontsize=15)
matplotlib.pyplot.xticks(numpy.arange(0, num_generations+1, 100), fontsize=10)
matplotlib.pyplot.yticks(numpy.arange(0, 101, 5), fontsize=10)
matplotlib.pyplot.show()


f = open("weights_" + str(num_generations) + "_iterations_" + str(mutation_percent) + "%_mutation.pkl", "wb")
pickle.dump(pop_weights_mat, f)
f.close()
