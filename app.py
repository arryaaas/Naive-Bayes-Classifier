from flask import Flask, render_template, request
from scripts import *

app = Flask(__name__)

data_train = data_training()
data_positif = data_grouping(data_train, "Positif")
data_negatif = data_grouping(data_train, "Negatif")

@app.route('/', methods=["POST", "GET"])
def training():
    title = "Data Training ( Data Latih )"
    data_train = data_training()
    return render_template(
        "training.html", 
        tables=[data_train.to_html(classes="data", header="true")], title=title
    )

@app.route('/grouping', methods=["POST", "GET"])
def grouping():
    title = "Data Grouping ( Pengelompokan Data )"
    total_positif = len(data_positif)
    total_negatif = len(data_negatif)
    return render_template(
        "grouping.html", 
        tables1=[data_positif.to_html(classes="data", header="true")], 
        tables2=[data_negatif.to_html(classes="data", header="true")], 
        title=title,
        total_positif=total_positif,
        total_negatif=total_negatif
    )

@app.route('/mean', methods=["POST", "GET"])
def mean():
    title = "Mean ( Nilai Rata-rata )"
    data_mean = df_mean(data_positif, data_negatif)
    return render_template(
        "dataframe.html", 
        tables=[data_mean.to_html(classes="data", header="true")], 
        title=title
    )

@app.route('/std', methods=["POST", "GET"])
def std():
    title = "Standar Deviasi ( Simpangan Baku )"
    data_std = df_std(data_positif, data_negatif)
    return render_template(
        "dataframe.html", 
        tables=[data_std.to_html(classes="data", header="true")], 
        title=title
    )

@app.route('/prob', methods=["POST", "GET"])
def prob():
    title = "Probabilitas ( Peluang )"
    data_prob = df_prob(data_train, data_positif, data_negatif)
    return render_template(
        "dataframe.html", 
        tables=[data_prob.to_html(classes="data", header="true")], 
        title=title
    )

@app.route('/testing')
def testing():
    title = "Data Testing ( Data Uji )"
    return render_template(
        "testing.html", 
        title=title
    )

@app.route('/predict', methods=["POST"])
def predict():
    title = "Predict ( Prediksi )"

    df_testing = pd.DataFrame({
        "Provinsi" : request.form.get("nama_provinsi"),
        "Dalam Perawatan" : int(request.form.get("dalam_perawatan")),
        "Sembuh" : int(request.form.get("sembuh")),
        "Meninggal" : int(request.form.get("meninggal"))
    }, index = [0])

    data_mean = df_mean(data_positif, data_negatif)
    data_std = df_std(data_positif, data_negatif)

    classification(df_testing, data_mean, data_std)

    return render_template(
        "dataframe.html", 
        tables=[df_testing.to_html(classes="data", header="true")], 
        title=title
    )

# if __name__ == '__main__':
#     app.run(debug=True)

app.run(debug=True)