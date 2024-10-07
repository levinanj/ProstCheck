from flask import Flask, render_template, request, redirect, url_for
from model import train_models, predict_diagnosis

app = Flask(__name__)

# Latih model
best_model, max_values,_,_ = train_models()

@app.route('/')
def home():
    title = "Aplikasi Flask"
    message = "Ini adalah contoh menggunakan Flask dan Jinja2."
    return render_template('index.html', title=title, message=message)


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        # Ambil data dari form
        nama = request.form['name']
        berat_badan = request.form['weight']
        tinggi_badan = request.form['height']
        umur = request.form['age']
        
        input_data = {
            'radius': float(request.form['radius']),
            'texture': float(request.form['texture']),
            'perimeter': float(request.form['perimeter']),
            'area': float(request.form['area']),
            'smoothness': float(request.form['smoothness']),
            'compactness': float(request.form['compactness']),
            'symmetry': float(request.form['symmetry']),
            'fractal_dimension': float(request.form['fractal_dimension'])
        }

        # Prediksi hasil diagnosis menggunakan model terbaik
        prediction_result = predict_diagnosis(best_model, input_data, max_values)

        # Interpretasi hasil prediksi
        # diagnosis = 'Malignant' if prediction_result == 1 else 'Benign'
        diagnosis = prediction_result
        # Redirect ke halaman hasil prediksi dengan membawa input data dan hasil diagnosis
        return redirect(url_for('hasil_prediksi', 
                                nama = nama,
                                berat_badan = berat_badan,
                                tinggi_badan = tinggi_badan,
                                umur = umur,
                                diagnosis=diagnosis, 
                                radius=input_data['radius'],
                                texture=input_data['texture'],
                                perimeter=input_data['perimeter'],
                                area=input_data['area'],
                                smoothness=input_data['smoothness'],
                                compactness=input_data['compactness'],
                                symmetry=input_data['symmetry'],
                                fractal_dimension=input_data['fractal_dimension']
                                ))
    
    return render_template('prediction.html')


# @app.route('/hasil_prediksi', methods=['GET'])
# def hasil_prediksi():
#     nama = request.args.get('nama')
#     berat_badan = request.args.get('berat_badan')
#     tinggi_badan = request.args.get('tinggi_badan')
#     umur = request.args.get('umur')
#     diagnosis = request.args.get('diagnosis')
#     radius = request.args.get('radius')
#     texture = request.args.get('texture')
#     perimeter = request.args.get('perimeter')
#     area = request.args.get('area')
#     smoothness = request.args.get('smoothness')
#     compactness = request.args.get('compactness')
#     symmetry = request.args.get('symmetry')
#     fractal_dimension = request.args.get('fractal_dimension')

#     return render_template('submit.html',
#                            nama = nama,
#                             berat_badan = berat_badan,
#                             tinggi_badan = tinggi_badan,
#                             umur = umur,
#                            diagnosis=diagnosis, 
#                            radius=radius, texture=texture, 
#                            perimeter=perimeter, area=area, 
#                            smoothness=smoothness, compactness=compactness, 
#                            symmetry=symmetry, fractal_dimension=fractal_dimension)

@app.route('/hasil_prediksi', methods=['GET'])
def hasil_prediksi():
    nama = request.args.get('nama')
    berat_badan = request.args.get('berat_badan')
    tinggi_badan = request.args.get('tinggi_badan')
    umur = request.args.get('umur')
    diagnosis = request.args.get('diagnosis')
    radius = request.args.get('radius')
    texture = request.args.get('texture')
    perimeter = request.args.get('perimeter')
    area = request.args.get('area')
    smoothness = request.args.get('smoothness')
    compactness = request.args.get('compactness')
    symmetry = request.args.get('symmetry')
    fractal_dimension = request.args.get('fractal_dimension')

    # Ambil informasi model dan metrik dari hasil training
    _, _, best_model_name, metrics_results = train_models()

    return render_template('submit.html',
                           nama=nama,
                           berat_badan=berat_badan,
                           tinggi_badan=tinggi_badan,
                           umur=umur,
                           diagnosis=diagnosis, 
                           radius=radius, texture=texture, 
                           perimeter=perimeter, area=area, 
                           smoothness=smoothness, compactness=compactness, 
                           symmetry=symmetry, fractal_dimension=fractal_dimension,
                           best_model_name=best_model_name,
                           metrics_results=metrics_results)



if __name__ == '__main__':
    app.run(debug=True)
