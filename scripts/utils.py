import pandas as pd
import math

def data_training():
    return pd.read_csv("data/covid19.csv", sep=";")

def data_grouping(df_training, category):
    return df_training[df_training["Kasus Terbesar Per Provinsi"] == category]

def mean(data):
    return sum(data) / len(data)

def stdev(data):
    n = len(data)
    squares = [item**2 for item in data]
    variance = ((n*sum(squares)) - (sum(data)**2)) / (n*(n-1))
    return variance**0.5

def df_mean(df_positif, df_negatif):
    return pd.DataFrame({
        "Kasus Terbesar Per Provinsi" : ["Positif", "Negatif"],
        "Dalam Perawatan" : [mean(df_positif["Dalam Perawatan"]),  mean(df_negatif["Dalam Perawatan"])],
        "Sembuh" : [mean(df_positif["Sembuh"]), mean(df_negatif["Sembuh"])],
        "Meninggal" : [mean(df_positif["Meninggal"]), mean(df_negatif["Meninggal"])]
    })

def df_std(df_positif, df_negatif):
    return pd.DataFrame({
        "Kasus Terbesar Per Provinsi" : ["Positif", "Negatif"],
        "Dalam Perawatan" : [stdev(df_positif["Dalam Perawatan"]), stdev(df_negatif["Dalam Perawatan"])],
        "Sembuh" : [stdev(df_positif["Sembuh"]), stdev(df_negatif["Sembuh"])],
        "Meninggal" : [stdev(df_positif["Meninggal"]), stdev(df_negatif["Meninggal"])]
    })

def df_prob(df_training, df_positif, df_negatif):
    return pd.DataFrame({
        "Kasus Terbesar Per Provinsi" : ["Positif", "Negatif"],
        "Nilai Probabilitas" : [len(df_positif) / len(df_training), len(df_negatif) / len(df_training)]
    })

def densitas_gauss(x, mean, std):
    return (1 / math.sqrt(2 * 3.14 * std)) * math.exp(-1 * (((x - mean) ** 2) / (2 * std ** 2)))

def classification(data, df_mean, df_std):
    category = []
    
    for i in range(len(data)):
        densitas_gauss_positif = [
            # Probabilitas dalam perawatan (positif)
            densitas_gauss(data.iloc[i][1], df_mean.iloc[0][1], df_std.iloc[0][1]),
            # Probabilitas sembuh (positif)
            densitas_gauss(data.iloc[i][2], df_mean.iloc[0][2], df_std.iloc[0][2]),
            # Probabilitas meninggal (positif)
            densitas_gauss(data.iloc[i][3], df_mean.iloc[0][3], df_std.iloc[0][3])
        ]
        
        densitas_gauss_negatif = [
            # Probabilitas dalam perawatan (negatif)
            densitas_gauss(data.iloc[i][1], df_mean.iloc[1][1], df_std.iloc[1][1]),
            # Probabilitas sembuh (positif)
            densitas_gauss(data.iloc[i][2], df_mean.iloc[1][2], df_std.iloc[1][2]),
            # Probabilitas meninggal (positif)
            densitas_gauss(data.iloc[i][3], df_mean.iloc[1][3], df_std.iloc[1][3])
        ]

        posterior = [
            # Probabilitas dalam perawatan * probabilitas sembuh * probabilitas meninggal
            densitas_gauss_positif[0] * densitas_gauss_positif[1] * densitas_gauss_positif[2],
            densitas_gauss_negatif[0] * densitas_gauss_negatif[1] * densitas_gauss_negatif[2]
        ]
        
        if max(posterior) == posterior[0]:
            category.append("Positif")
        else:
            category.append("Negatif")
    
    data["Kasus Terbesar Per Provinsi (Predict)"] = category

def accuracy(data):
    success, failed = 0, 0

    for i in range(len(data)):
        if (data.iloc[i][4] == data.iloc[i][5]):
            success += 1
        else :
            failed +=1

    return success, failed
