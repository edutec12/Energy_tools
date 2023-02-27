# import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
import copy
from typing import Any
from math import e
import math as math
from sklearn.cluster import MiniBatchKMeans

# read data from Excel files
df = pd.read_excel('full_data.xlsx')  # read data for consumption
tm = pd.read_excel('temperatura_amb.xlsx')  # read data for temperature

# convert time columns to datetime format
df['tiempo'] = pd.to_datetime(df['tiempo'])
tm['tiempot'] = pd.to_datetime(tm['tiempot'])

# select data for demand and active maximum with a given series number
us = df[(df['variable'] == 'demandaactivamax') & (df['Serie'] == 134874)]

# group data by hour and calculate the mean value
us = us.groupby(pd.Grouper(key='tiempo', freq='H')).mean().fillna(0).reset_index()
tm = tm.groupby(pd.Grouper(key='tiempot', freq='H')).mean().fillna(0).reset_index()

# resample data with a frequency of 1 hour and fill in missing values
us = us.set_index('tiempo').resample('H').ffill().reset_index()
tm = tm.set_index('tiempot').resample('H').ffill().reset_index()

# plot the annual consumption and temperature
plt.plot(us.valor)  # plot annual consumption
plt.title('Annual Consumption [kWh]')  # add title to the plot
plt.show()  # display the plot

plt.plot(tm.temp)  # plot annual temperature
plt.title('Annual Temperature [ºC]')  # add title to the plot
plt.show()  # display the plot

# Compute means for each day of the week
prome = us.valor.mean()
promet = tm.temp.mean()
dw = us['tiempo'].dt.dayofweek
dtem = tm['tiempot'].dt.dayofweek

def mean(day_number, data):
    selection = data[dw == day_number]
    if len(selection) == 0:
        selection = data[dw == day_number - 1]
    return selection['valor'].mean()

def meant(day_number):
    cont = 0
    sum = 0
    index_i: int
    for index_i in range(len(tm)):
        if dtem[index_i] == day_number and tm.temp[index_i] != 0:
            cont += 1
            sum = sum + tm.temp[index_i]
    if cont == 0:
        exit()
    prom = sum / cont
    if prom == 0:
        cont = 0
        sum = 0
        for index_i in range(len(tm)):
            if dtem[index_i] == day_number - 1 and tm.temp[index_i] != 0:
                cont += 1
                sum = sum + tm.valor[index_i]
    return sum / cont


promet = np.mean(tm.temp)
    

def fullu(day, us):
    us = us.copy()  # make a copy of the DataFrame
    for i_idx in range(len(us)):
        if us.valor[i_idx] < 1 and dw[i_idx] == day:
            us.valor[i_idx] = mean(day, us)
    return us.valor

def fullt(day):
    tm_new = tm.copy()  # make a copy of the DataFrame
    for i_idx in range(len(tm)):
        if tm.temp[i_idx] == 0 and dtem[i_idx] == day:
            tm_new.temp[i_idx] = meant(day)
        if tm.temp[i_idx] < 1 and dtem[i_idx] == day:
            tm_new.temp[i_idx] = meant(day)
    return tm_new.temp

for i in range(7):
    us.valor = fullu(i, us)
    tm.temp = fullt(i)

# Plot resampled consumption and temperature
x_horas = us['tiempo']
y_power = us['valor']
plt.plot(x_horas, y_power)
plt.title('Consumo Anual [kWh]')
plt.show()

x_horast = tm['tiempot']
y_tiempot = tm['temp']
plt.plot(x_horast, y_tiempot)
plt.title('Consumo Anual [kWh]')
plt.show()

# Round timestamps to the nearest hour
us['tiempo'] = pd.to_datetime(us['tiempo']).dt.round('H')
tm['tiempot'] = pd.to_datetime(tm['tiempot']).dt.round('H')

# Drop unnecessary columns and print dataframes
us_1 = us.drop(columns=['tiempo'])
tm_1 = tm.drop(columns=['tiempot'])
#print(us_1)
#print(tm_1)

# Create a new plot
figura, grafica = plt.subplots()

# Loop through the index of diaIndice minus 1
# diaIndice is assumed to be a list or array of integers
# representing the indices at which each new day begins
for j in range(len(diaIndice) - 1):
    i0 = diaIndice[j] # starting index for this day
    i1 = diaIndice[j + 1] # ending index for this day
    x_horas = us['tiempo'][i0:i1] - pd.Timedelta(days=j) # x-axis values for this day (adjusted for number of days)
    y_ener = us['valor'][i0:i1] # y-axis values for this day
    graf = grafica.plot(x_horas, y_ener, alpha=0.5) # plot the data for this day with transparency

# Format the x-axis tick labels to show only the hour and minute
formato = dates.DateFormatter("%H:%M")
grafica.xaxis.set_major_formatter(formato)

# Add a title to the plot and display it
plt.title('valor')
plt.show()

# Print the contents of the 'us' and 'tm' DataFrames to the console
#print(us)
#print(tm)

# Merge the 'us' and 'tm' DataFrames based on the 'tiempot' and 'tiempo' columns, respectively
# Then drop the 'tiempot' column from the resulting DataFrame
pivo = pd.merge(tm, us, left_on='tiempot', right_on='tiempo', how='outer').drop(columns=['tiempot'])

# Extract the hour and date from the 'tiempo' column and add them as new columns to the DataFrame
pivo["hora"] = pivo["tiempo"].dt.hour
pivo["fecha"] = pivo["tiempo"].dt.date

# Drop any rows that contain missing values (NaN)
pivo = pivo.dropna()

# Print the resulting DataFrame to the console
#print(pivo)

# Plot the 'valor' column of the DataFrame against the 'tiempo' column (time series)
plt.plot(pivo['tiempo'], pivo['valor'])
plt.title('Consumo Anual [kWh]')
plt.show()

# Plot the 'temp' column of the DataFrame against the 'tiempo' column (time series)
plt.plot(pivo['tiempo'], pivo['temp'])
plt.title('Consumo Anual [kWh]')
plt.show()

# Make a copy of the DataFrame and remove some unnecessary columns
pivoted=copy.copy(pivo)
pivoted.pop("tiempo")
pivoted.pop("id")
pivoted.pop("Serie")
pivoted.pop("Medida")


# set the "hora" column as the index
pivoted = pivoted.set_index("hora")
#print(pivoted)
pivote_us=copy.copy(pivoted.valor)
#print(pivo)


# Pivot the dataframe with datetime as index, hour as columns and energy value as values
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score


def plot_energy_clusters(pivo,value):
    # Pivot the DataFrame
    pivo = pivo.pivot(index='fecha', columns='hora', values=value)
    # Remove rows with NaN values
    pivo1 = pivo.dropna()

    # Scale the matrix to range [0,1]
    sc = MinMaxScaler()
    X = sc.fit_transform(pivo1)

    # Initialize a list to store silhouette scores for each cluster
    silhouette_scores = []
    # Create a range of cluster numbers to test
    n_clusters = np.arange(2, 31).astype(int)

    # For each number of clusters in the range, perform KMeans clustering and store the silhouette score
    for n in n_clusters:
        kmeans = KMeans(n_clusters=n)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))

    # Plot the silhouette scores for each cluster
    plt.plot(silhouette_scores)
    plt.title('Silhouette Scores for KMeans Clustering')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    # Perform KMeans clustering on the matrix X with 3 clusters
    kmeans = MiniBatchKMeans(n_clusters=3)
    labels = kmeans.fit_predict(X)
    # Create a Pandas series with the cluster labels and append to the original dataframe
    clusters = pd.Series(labels, name="cluster")
    pivo1 = pivo1.set_index(clusters, append=True)

    # Plot each cluster with its median hourly energy usage
    fig, ax = plt.subplots(1, 1, figsize=(13,8))
    color_list = ['yellow', 'red', 'orange', 'magenta', 'black', 'blue']
    cluster_values = sorted(pivo1.index.get_level_values('cluster').unique())

    for cluster, color in zip(cluster_values, color_list):
        pivo1.xs(cluster, level=1).T.plot(
            ax=ax, legend=False, alpha=0.1, color=color, label=f'Cluster {cluster}'
        )
        pivo1.xs(cluster, level=1).median().plot(
            ax=ax, legend=False, color=color, alpha=0.9
        )

    # Set the x-ticks, x and y-labels, and title of the plot
    ax.set_xticks(np.arange(1, 25))
    ax.set_ylabel('medida')
    ax.set_xlabel('hora')
    plt.title('Cluster Plots with Median Hourly Usage')
    plt.show()

plot_energy_clusters(pivo,'valor')

plot_energy_clusters(pivo,'temp')


# Pivot the DataFrame to create a matrix of hourly energy production
coinc = pivo.pivot(index="fecha", columns="tiempo", values="valor")

# Calculate the total energy production per hour and plot
sum_hourly = coinc.sum(axis=0)
plt.plot(sum_hourly, label='Total Energy Production')

# Calculate the energy peak per hour and plot
peak_hourly = coinc.max(axis=0)
plt.plot(peak_hourly, label="Peak Energy Production")

# Calculate the energy coincidence factor per hour and plot
coincidence_factor = peak_hourly / sum_hourly
plt.plot(coincidence_factor, label="Energy Coincidence Factor")

# Calculate the diversity factor per hour and plot
diversity_factor = 1 - (coinc.std(axis=0) / coinc.mean(axis=0))
plt.plot(diversity_factor, label="Energy Diversity Factor")

# Set the x-label, y-label, and title of the plot
plt.xlabel('Hour of the Day')
plt.ylabel('Energy (kW)')
plt.title('Hourly Energy Production, Peak Energy, Coincidence Factor, and Diversity Factor')
plt.legend()
plt.show()

# Print the total energy production and peak energy values
total_energy = sum_hourly.sum()
peak_energy = peak_hourly.max()
#print("Total Energy Production: ", total_energy)
#print("Peak Energy Production: ", peak_energy)

# Print the energy coincidence factor and diversity factor
coincidence_factor_value = peak_energy / total_energy
diversity_factor_value = diversity_factor.mean()
#print("Energy Coincidence Factor: ", coincidence_factor_value)
#print("Energy Diversity Factor: ", diversity_factor_value)



coinc=copy.copy(pivo)

coinc=coinc.pivot(index="fecha", columns="tiempo", values="valor")


idx=coinc.index
coinc_top = coinc.columns

suma = coinc.sum(axis = 1)
#print(suma)
plt.plot(suma, label='energía')

suma_total = suma.values.sum()

#print(suma_total)


pico=coinc.max(axis=1)
plt.plot(pico,label="Pico")
#print(pico)
plt.legend()
plt.show()

horapico: float = coinc.idxmax(axis=1)


#print(pivo)
power=us.valor

factores=pd.concat([suma, pico,suma-pico],axis=1)
#print(factores)


#print(factores)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import copy

# Define the input matrix X and target vector y
X = factores.values
target = factores.index.to_numpy()

# Perform KMeans clustering with 3 clusters and a bad initialization
kmeans = KMeans(n_clusters=3, n_init=1, init="random")
kmeans.fit(X)
labels = kmeans.labels_

# Plot the clusters in a 3D plot with peak hour, energy peak, and coincidence factor as the three axes
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
ax.scatter(X[:, 2], X[:, 0], X[:, 1], c=labels.astype(float), edgecolor="k")
ax.set_xlabel("Peak hour")
ax.set_ylabel("Energy peak")
ax.set_zlabel("CF")
ax.set_title("Cluster")
ax.dist = 12

# Plot the ground truth
for name, label in [("Consumer behaviour 1", 0), ("Consumer behaviour 2", 1), ("Consumer behaviour 3", 2)]:
    ax.text3D(
        X[target == label, 2].mean(),
        X[target == label, 0].mean(),
        X[target == label, 1].mean() + 2,
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.2, edgecolor="w", facecolor="w"),
    )

# Show the plot
plt.show()



