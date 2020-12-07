# Copyright [2020] [Quantum-Chain]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dwave_qbsolv import QBSolv
from dwave.system import LeapHybridSampler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import pandas as pd
import geopandas
from sklearn.cluster import KMeans

import math
import random

import dwavebinarycsp
import dwave.inspector
from dwave.system import EmbeddingComposite, DWaveSampler

from utilities import get_groupings, visualize_groupings, visualize_scatterplot, visualize_groupings_again

Total_Number_Cities = 40
Number_Deliveries = 8
#cd = (int)(Total_Number_Cities/Number_Deliveries)

# Tunable parameters. 
A = 9000
B = 1
chainstrength = 1
#numruns = 100

## Clustering Preprocess

def lat_lon_distance(a, b):
    """Calculates distance between two latitude-longitude coordinates."""
    R = 3963  # radius of Earth (miles)
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    return math.acos(math.sin(lat1) * math.sin(lat2) +
                     math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2)) * R


def clustering(scattered_points,filename):
    kmeans = KMeans(n_clusters = 8, random_state = 42, init = 'k-means++', n_init = 10, max_iter=30, algorithm='full').fit(scattered_points)
    groupings = {}
    dist_from_cent = {}

    for i in range(Number_Deliveries):
        groupings[str(i)] = []
        dist_from_cent[str(i)] = []

    for i in range(len(scattered_points)):
        for key in groupings.keys(): 
            if str(kmeans.labels_[i]) == key:
                groupings[key].append(scattered_points[i])
                

    #print(groupings)

    visualize_groupings(groupings, filename)

    return groupings

def plot_map(route,cities, cities_index, cities_lookup,filename):

    data_list=[[key, cities[key][0], - cities[key][1]] for key in cities.keys()]
    df = pd.DataFrame(data_list)
    data_list=[[cities_lookup[cities_index[route[i]]], cities[cities_lookup[cities_index[route[i]]]][0], - cities[cities_lookup[cities_index[route[i]]]][1]] for i in range(len(route))]
    df_visit = pd.DataFrame(data_list)
    
    #City,Latitude,Longitude
    df.columns=['City','Latitude','Longitude']
    df_visit.columns = ['City','Latitude','Longitude']
    df_start = df_visit[df_visit['City'].isin([cities_lookup[cities_index[route[0]]]])]  
    df_end = df_visit[df_visit['City'].isin([cities_lookup[cities_index[route[len(route)-1]]]])]

    gdf_all = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))
    gdf_visit = geopandas.GeoDataFrame(
        df_visit, geometry=geopandas.points_from_xy(df_visit.Longitude, df_visit.Latitude))
    gdf_start = geopandas.GeoDataFrame(
        df_start, geometry=geopandas.points_from_xy(df_start.Longitude, df_start.Latitude))
    gdf_end = geopandas.GeoDataFrame(
        df_end, geometry=geopandas.points_from_xy(df_end.Longitude, df_end.Latitude))

    world = geopandas.read_file(
            geopandas.datasets.get_path('naturalearth_lowres'))

    # Restrict to the USA only.
    ax = world[world.name == 'United States of America'].plot(
        color='white', edgecolor='black')

    # plot the ``GeoDataFrame``
    x_values=gdf_visit.values.T[2]
    y_values=gdf_visit.values.T[1]
    plt.plot(x_values,y_values)

    gdf_all.plot(ax=ax, color='gray')
    gdf_visit.plot(ax=ax, color='blue')
    gdf_start.plot(ax=ax, color='green')
    gdf_end.plot(ax=ax, color='red')

    ax.set_xlim(xmin=-130, xmax=-65)
    ax.set_ylim(ymin=20, ymax=55)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_aspect(1.2)

    ax.legend(['Path','All cites', 'To Visit','Start','End'])

    plt.savefig(filename)
    #plt.show()

cities = {
        'New York City': (40.72, 74.00),
        'Los Angeles': (34.05, 118.25),
        'Chicago': (41.88, 87.63),
        'Houston': (29.77, 95.38),
        'Phoenix': (33.45, 112.07),
        'Philadelphia': (39.95, 75.17),
        'San Antonio': (29.53, 98.47),
        'Dallas': (32.78, 96.80),
        'San Diego': (32.78, 117.15),
        'San Jose': (37.30, 121.87),
        'Detroit': (42.33, 83.05),
        'San Francisco': (37.78, 122.42),
        'Jacksonville': (30.32, 81.70),
        'Indianapolis': (39.78, 86.15),
        'Austin': (30.27, 97.77),
        'Columbus': (39.98, 82.98),
        'Fort Worth': (32.75, 97.33),
        'Charlotte': (35.23, 80.85),
        'Memphis': (35.12, 89.97),
        'Baltimore': (39.28, 76.62),
        'Las Vegas': (36.17, 115.14),
        'Washington DC': (38.91, 77.04),
        'Seattle': (47.61, 122.33),
        'Miami': (25.76, 80.19),
        'Denver': (39.74, 104.99),
        'Nashville': (36.16, 86.78),
        'Minneapolis': (44.98, 93.27),
        'Oakland': (37.80, 122.27),
        'St. Louis': (28.63, 90.20),
        'Sacramento': (38.58, 121.49),
        'Albuquerque': (35.08, 106.65),
        'Tucson': (32.22, 110.97),
        'Cincinnati': (39.10, 84.51),
        'Tampa': (27.95, 82.46),
        'Boston': (42.36, 71.06),
        'Portland': (45.51, 122.68),
        'Pittsburgh': (40.44, 80.00),
        'Omaha': (41.26, 95.93),
        'Tulsa': (36.15, 95.99),
        'El Paso': (31.76, 106.49),
    }

cities_lookup = {
        0: 'New York City',
        1: 'Los Angeles',
        2: 'Chicago',
        3: 'Houston',
        4: 'Phoenix',
        5: 'Philadelphia',
        6: 'San Antonio',
        7: 'Dallas',
        8: 'San Diego',
        9: 'San Jose',
        10: 'Detroit',
        11: 'San Francisco',
        12: 'Jacksonville',
        13: 'Indianapolis',
        14: 'Austin',
        15: 'Columbus',
        16: 'Fort Worth',
        17: 'Charlotte',
        18: 'Memphis',
        19: 'Baltimore',
        20: 'Las Vegas',
        21: 'Washington DC',
        22: 'Seattle',
        23: 'Miami',
        24: 'Denver',
        25: 'Nashville',
        26: 'Minneapolis',
        27: 'Oakland',
        28: 'St. Louis',
        29: 'Sacramento',
        30: 'Albuquerque',
        31: 'Tucson',
        32: 'Cincinnati',
        33: 'Tampa',
        34: 'Boston',
        35: 'Portland',
        36: 'Pittsburgh',
        37: 'Omaha',
        38: 'Tulsa',
        39: 'El Paso'
    }

cities_index = {
        (40.72, 74.00) : 0,
        (34.05, 118.25) : 1,
        (41.88, 87.63) : 2,
        (29.77, 95.38) : 3,
        (33.45, 112.07): 4,
        (39.95, 75.17): 5,
        (29.53, 98.47): 6,
        (32.78, 96.80): 7,
        (32.78, 117.15): 8,
        (37.30, 121.87): 9,
        (42.33, 83.05): 10,
        (37.78, 122.42): 11,
        (30.32, 81.70): 12,
        (39.78, 86.15): 13,
        (30.27, 97.77): 14,
        (39.98, 82.98): 15,
        (32.75, 97.33): 16,
        (35.23, 80.85): 17,
        (35.12, 89.97): 18,
        (39.28, 76.62): 19,
        (36.17, 115.14): 20,
        (38.91, 77.04): 21,
        (47.61, 122.33): 22,
        (25.76, 80.19): 23,
        (39.74, 104.99): 24,
        (36.16, 86.78): 25,
        (44.98, 93.27): 26,
        (37.80, 122.27): 27,
        (28.63, 90.20): 28,
        (38.58, 121.49): 29,
        (35.08, 106.65): 30,
        (32.22, 110.97): 31,
        (39.10, 84.51): 32,
        (27.95, 82.46): 33,
        (42.36, 71.06): 34,
        (45.51, 122.68): 35,
        (40.44, 80.00): 36,
        (41.26, 95.93): 37,
        (36.15, 95.99): 38,
        (31.76, 106.49): 39,
    }

    # initial state, a randomly-ordered itinerary
init_state = list(cities.values())
random.shuffle(init_state)

clustered_filename = "twentyone_cities_clustered.png"
citygroups = clustering(init_state, clustered_filename)
citygroup_count = 0

for clust, points in citygroups.items():
        
        # Ignore items that do not contain any coordinates
        if not points:
            continue

        pcount = 0

        points_array = np.array(points)
        points_len = len(points_array)
       

        D = [[0 for z in range(Total_Number_Cities)] for y in range(Total_Number_Cities)]

        for i in range(len(points_array)-1):
            for j in range(i+1, len(points_array)):
            #if(i+1 < len(points_array)):
                first_city = points_array[i]
                second_city = points_array[j]
                citya = tuple(first_city.tolist())
                cityb = tuple(second_city.tolist())
                namea = cities_index[citya]
                nameb = cities_index[cityb]
                D[namea][nameb] = D[nameb][namea] = lat_lon_distance(citya,cityb)
                


        # Function to compute index in QUBO for variable 
        def return_QUBO_Index(a, b):
            return (a)*len(citygroups[clust])+(b)

        print("size of dict[clust]:",len(citygroups[clust]))

        ## Creating the QUBO
        # Start with an empty QUBO
        
        Q = {}
        for i in range(len(citygroups[clust])*len(citygroups[clust])):
            for j in range(len(citygroups[clust])*len(citygroups[clust])):
                Q.update({(i,j): 0})

        # Constraint that each row has exactly one 1, constant = N*A
        for v in range(len(citygroups[clust])):
            for j in range(len(citygroups[clust])):
                Q[(return_QUBO_Index(v,j), return_QUBO_Index(v,j))] += -1*A
                for k in range(j+1, len(citygroups[clust])):
                    Q[(return_QUBO_Index(v,j), return_QUBO_Index(v,k))] += 2*A
                    Q[(return_QUBO_Index(v,k), return_QUBO_Index(v,j))] += 2*A

        # Constraint that each col has exactly one 1
        for j in range(len(citygroups[clust])):
            for v in range(len(citygroups[clust])):
                Q[(return_QUBO_Index(v,j), return_QUBO_Index(v,j))] += -1*A
                for w in range(v+1,len(citygroups[clust])):
                    Q[(return_QUBO_Index(v,j), return_QUBO_Index(w,j))] += 2*A
                    Q[(return_QUBO_Index(w,j), return_QUBO_Index(v,j))] += 2*A

        # Objective that minimizes distance
        for u in range(len(citygroups[clust])):
            for v in range(len(citygroups[clust])):
                if u!=v:
                    for j in range(len(citygroups[clust])):
                        Q[(return_QUBO_Index(u,j), return_QUBO_Index(v,(j+1)%len(citygroups[clust])))] += B*D[u][v]

        # Run the QUBO using qbsolv (classically solving)
        resp = QBSolv().sample_qubo(Q)

        # Use LeapHybridSampler() for faster QPU access
        #sampler = LeapHybridSampler()
        #resp = sampler.sample_qubo(Q)

        # First solution is the lowest energy solution found
        sample = next(iter(resp))

        sample1 = iter(resp)
        #print('{}'.format(', '.join(map(str, sample))))


        #print("This is the sample:",sample)

        # Display energy for best solution found
        print('Energy: ', next(iter(resp.data())).energy)

        # Print route for solution found
        route = [-1]*len(citygroups[clust])
        for node in sample:
            if sample[node]>0:
                j = node%len(citygroups[clust])
                v = (node-j)/len(citygroups[clust])
                route[j] = int(v)
        
        
        

        # Compute and display total mileage
        mileage = 0
        for i in range(len(citygroups[clust])-1):
            first_city = points_array[route[i]]
            second_city = points_array[route[i+1]]
            citya = tuple(first_city.tolist())
            cityb = tuple(second_city.tolist())
            namea = cities_index[citya]
            nameb = cities_index[cityb]
            mileage+= D[namea][nameb]
            
        print('Mileage: ', mileage)

        filename = "Hackathon_Route_Map_" + str(citygroup_count)
        citygroup_count = citygroup_count + 1

        #print([tuple(points_array[route[i]]) for i in range(len(route))])
        plot_map([tuple(points_array[route[i]]) for i in range(len(route))],cities, cities_index, cities_lookup, filename)

        # Print route:

        for i in range(len(route)):
            print(cities_lookup[cities_index[tuple(points_array[route[i]])]]+'\n')
      
