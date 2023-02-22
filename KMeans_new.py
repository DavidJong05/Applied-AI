from sklearn import preprocessing
from matplotlib import pyplot as plt
import numpy as np
import math as m

data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
#v_data = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
#v_dates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
days = np.genfromtxt('days.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
centroids = np.genfromtxt('centroids.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})



def Kmeans(data, centroids):
    '''' This function returns the average squared distance from the data points to their shortest distance centroid.
    Function also calculates the best centered centroids to cluster the data'''
    normalized_data = preprocessing.normalize(data)
    normalized_centroids = preprocessing.normalize(centroids)
    shortest_distance = m.inf
    shortest_c = np.array([])
    cluster_dict = {} # Dictionary with centroids as keys, and datapoints as values
    V = 0 # V = xi -> cj (cj - xi)^2  for avg sqrt dist
    amount_of_points = 0

    for centroid in normalized_centroids: # Initialize each centroid as a key to cluster_dict
        cluster_dict[np.array2string(centroid)] = []

    for d_point in normalized_data:
        for c_point in normalized_centroids: # for loop compares each centroid to the given data point, to calc shortest distance
            distance = np.linalg.norm(d_point - c_point) #euclidian
            if distance < shortest_distance:
                shortest_distance = distance # Closest centroid from data point
                shortest_c = np.array2string(c_point) # Has to match the key in dict which is a string
        V += shortest_distance * shortest_distance # V = xi -> cj (cj - xi)^2
        amount_of_points += 1 # to calc avg
        cluster_dict[shortest_c].append([d_point]) # Add the data point to the centroid key he is closest to
        shortest_distance = m.inf

    average_centroid_dist = V/amount_of_points
    mean_arr = []
    list_of_means = []
    for key in cluster_dict.keys():
        for value in cluster_dict[key]:
            mean_arr.append(value[0]) # Add each data point to mean_arr from the given key
        mean_arr = np.mean(mean_arr,axis=0) # Mean of all data points of a single dict key centroid
        list_of_means.append(mean_arr)
        mean_arr = []
    new_centroids = np.array(list_of_means, dtype=object) # Convert to np.ndarray

    if np.array_equal(new_centroids, centroids): # Done, there is no better midpoint for each centroid
        #for key in cluster_dict: # print each key with each assigned datapoint
        #    print("Best centroid: ",key, "\n", "Datapoints:")
        #    for value in cluster_dict[key]:
        #        print(value[0])
        #    print("--------------------------------------------", "\n")
        #print("Average squared distance", average_centroid_dist)
        return average_centroid_dist
    return Kmeans(data,new_centroids)



meetpunten = []
for K_cluster in range(1,12): # 12 clusters
    meetpunt = Kmeans(data, days[0:K_cluster])
    print("With",K_cluster, "cluster(s) the Average sqrt distance : ", meetpunt)
    meetpunten.append(meetpunt)

plt.plot(meetpunten)
plt.xlabel("Number of clusters")
plt.ylabel("Distance to centroids")
plt.show()



