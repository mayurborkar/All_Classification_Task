from application_logging.logger import App_Logger
from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle


logger = App_Logger('logFiles/App.log')

app = Flask(__name__)

logger.info("INFO", 'Loading Pickle File ')
model = pickle.load(open('Scaler.pkl', 'rb'))
model2 = pickle.load(open('bagging.pkl', 'rb'))
logger.info("INFO", 'Pickle File Load For Prediction')


@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def home():
    """
    :Method Name : home
    :DESC : This Will Return The Home Page
    :return : render_template (index.html) page
    """
    try:
        logger.info('INFO', 'The Home Page Is Displayed')
        return render_template('index.html')

    except Exception as e:
        raise Exception(f"(home) - Could not find index.html Page \n" + str(e))


@app.route("/report", methods=['GET', 'POST'])
@cross_origin()
def report():
    """
    :Method Name : report
    :DESC : This Will Return The Feature Analysis Page
    :return : render_template (Report.html) page
    """
    try:
        logger.info('INFO', 'The Report Method Is Calling For Showing The Report')
        return render_template('Report.html')

    except Exception as e:
        raise Exception(f"(report) - Could not find report.html Page \n" + str(e))


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    """
        :Method Name : predict
        :DESC : This Will Return The Prediction of User Input
        :return : render_template (index.html) page With result
    """
    if request.method == 'POST':
        logger.info("INFO", 'Request Method : POST')

        try:
            Time = float(request.form['Time'])

            Acceleration_frontal = float(request.form['Acceleration_frontal'])

            Acceleration_vertical = float(request.form['Acceleration_vertical'])

            Acceleration_lateral = float(request.form['Acceleration_lateral'])

            RSSI = float(request.form['RSSI'])

            Antenna = request.form['Antenna']

            Phase = float(request.form['Phase'])

            Frequency = float(request.form['Frequency'])

            logger.info("INFO", 'All The Feature Is Selected By Value')

            value = model.transform([[Time, Acceleration_frontal, Acceleration_vertical,
                                    Acceleration_lateral, RSSI, Antenna, Phase, Frequency]])
            logger.info('INFO', 'Applying Standard Scaler Pickle File For The Prediction')

            logger.info('INFO', 'Predicting The Final Outputs')
            prediction = model2.predict(value)

            if prediction == 1:
                label = 'Sit On Bed'
            elif prediction == 2:
                label = 'Sit On Chair'
            elif prediction == 3:
                label = 'Lying'
            else:
                label = 'Ambulating'

            return render_template('index.html', prediction_text=" The Label Activity Is {}".format(label))

        except Exception as e:
            raise Exception(f"(predict) - Their Is Something Wrong About Predict \n" + str(e))

    else:
        logger.info('INFO', 'Post Method Is Not Selected')
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
