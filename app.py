from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Cargar el modelo
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    
    if request.method == 'POST':
        try:
            private     = request.form['private']
            room_board  = float(request.form['room_board'])
            grad_rate   = float(request.form['grad_rate'])
            expend      = float(request.form['expend'])
            perc_alumni = float(request.form['perc_alumni'])

            # Validaciones
            if private not in ['Yes', 'No']:
                raise ValueError("Tipo de universidad inválido")
            if not (0 <= grad_rate <= 100):
                raise ValueError("La tasa de graduación debe estar entre 0 y 100")
            if room_board <= 0 or expend <= 0 or perc_alumni < 0:
                raise ValueError("Los valores numéricos deben ser positivos")

            # Crear dataframe con variables originales
            input_data = pd.DataFrame([{
                'Private':     private,
                'Room.Board':  room_board,
                'Grad.Rate':   grad_rate,
                'Expend':      expend,
                'perc.alumni': perc_alumni
            }])

            # El pipeline aplica OrdinalEncoder + StandardScaler internamente
            pred = model.predict(input_data)[0]
            prediction = f"${pred:,.0f} USD"

        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = f"Error al procesar los datos: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=False)