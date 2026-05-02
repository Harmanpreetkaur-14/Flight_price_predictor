from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle, os, numpy as np

app = Flask(__name__)

# ─── DATABASE CONFIG ───────────────────────────────────────────────────────────
# Local SQLite (change to PostgreSQL for production):
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/flightdb'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ─── DATABASE MODEL ────────────────────────────────────────────────────────────
class Prediction(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    airline       = db.Column(db.String(50))
    source_city   = db.Column(db.String(50))
    dest_city     = db.Column(db.String(50))
    travel_class  = db.Column(db.String(20))
    stops         = db.Column(db.Integer)
    departure_time= db.Column(db.String(30))
    arrival_time  = db.Column(db.String(30))
    duration      = db.Column(db.Float)
    days_left     = db.Column(db.Integer)
    predicted_price = db.Column(db.Float)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# ─── FEATURE ENCODING (mirrors your notebook preprocessing) ────────────────────
AIRLINES      = ['Air_India','AirAsia','GO_FIRST','Indigo','SpiceJet','Vistara']
CITIES        = ['Bangalore','Chennai','Delhi','Hyderabad','Kolkata','Mumbai']
TIME_SLOTS    = ['Afternoon','Early_Morning','Evening','Late_Night','Morning','Night']
STOPS_MAP     = {'zero': 0, 'one': 1, 'two_or_more': 2}

def build_feature_vector(data):
    """Encode input exactly as your notebook's one-hot + factorize pipeline."""
    feats = {}
    feats['class']    = 1 if data['travel_class'] == 'Business' else 0
    feats['stops']    = STOPS_MAP.get(data['stops'], 0)
    feats['duration'] = float(data['duration'])
    feats['days_left']= int(data['days_left'])

    for a in AIRLINES:
        feats[f'airline_{a}'] = 1 if data['airline'].replace(' ','_') == a else 0
    for c in CITIES:
        feats[f'source_city_{c}'] = 1 if data['source_city'] == c else 0
    for c in CITIES:
        feats[f'destination_city_{c}'] = 1 if data['dest_city'] == c else 0
    for t in TIME_SLOTS:
        feats[f'departure_time_{t}'] = 1 if data['departure_time'].replace(' ','_') == t else 0
    for t in TIME_SLOTS:
        feats[f'arrival_time_{t}'] = 1 if data['arrival_time'].replace(' ','_') == t else 0

    # Must match the exact column order used during model training
    col_order = (
        ['class','stops','duration','days_left'] +
        [f'airline_{a}' for a in AIRLINES] +
        [f'source_city_{c}' for c in CITIES] +
        [f'destination_city_{c}' for c in CITIES] +
        [f'departure_time_{t}' for t in TIME_SLOTS] +
        [f'arrival_time_{t}' for t in TIME_SLOTS]
    )
    return np.array([[feats[c] for c in col_order]])

# ─── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if data['source_city'] == data['dest_city']:
            return jsonify({'error': 'Source and destination cannot be the same.'}), 400

        if model:
            X = build_feature_vector(data)
            price = float(model.predict(X)[0])
        else:
            # Fallback demo logic when model.pkl is not yet present
            base = {'Indigo':3800,'Air India':5200,'SpiceJet':3500,
                    'Vistara':5800,'GO FIRST':3700,'AirAsia':3600}
            price = base.get(data['airline'], 4000)
            price *= 2.8 if data['travel_class'] == 'Business' else 1.0
            price += STOPS_MAP.get(data['stops'], 0) * 800
            price += float(data['duration']) * 400
            days  = int(data['days_left'])
            price *= 1.35 if days < 7 else (1.15 if days < 15 else (0.85 if days > 60 else 1.0))

        price = round(price, 2)

        # Save to DB
        rec = Prediction(
            airline=data['airline'], source_city=data['source_city'],
            dest_city=data['dest_city'], travel_class=data['travel_class'],
            stops=STOPS_MAP.get(data['stops'], 0),
            departure_time=data['departure_time'], arrival_time=data['arrival_time'],
            duration=float(data['duration']), days_left=int(data['days_left']),
            predicted_price=price
        )
        db.session.add(rec)
        db.session.commit()

        return jsonify({'price': price, 'id': rec.id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
    return jsonify([{
        'id': r.id, 'airline': r.airline,
        'from': r.source_city, 'to': r.dest_city,
        'class': r.travel_class, 'price': r.predicted_price,
        'date': r.created_at.strftime('%d %b %Y %H:%M')
    } for r in rows])

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
