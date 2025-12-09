from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from bson.json_util import dumps
from bson.objectid import ObjectId

app = Flask(__name__)
CORS(app)
app.config["MONGO_URI"] = "mongodb+srv://dbuser:abcd1234@cluster0.kf3ooia.mongodb.net/vehicle_data"
mongo = PyMongo(app)

@app.route("/api/v1/vehicles/<int:vehicle_id>/telemetry/latest", methods=['GET'])
def get_data(vehicle_id):
    data = mongo.db.telemetry.find_one({"vehicle_id": vehicle_id})
    if data:
        return dumps(data), 200
    return jsonify({"message": "No telemetry found"}), 404

@app.route("/api/v1/vehicles/telemetry/latest",methods=['POST'])
def post_data():
    body = request.get_json()
    inserted_id = mongo.db.telemetry.insert_one(body).inserted_id
    return jsonify({"message": "data inserted", "id": str(inserted_id)}), 201

@app.route('/api/v1/vehicles/<int:vehicle_id>/anomalies', methods=['GET'])
def get_anomaly_data(vehicle_id):
    data = mongo.db.anamolies.find({"vehicle_id": vehicle_id})
    return dumps(list(data)), 200


@app.route('/api/v1/anomalies', methods=["POST"])
def post_anomalies():
    body = request.get_json()
    inserted_id = mongo.db.anamolies.insert_one(body).inserted_id
    return jsonify({"message": "Anomaly inserted", "id": str(inserted_id)}), 201


@app.route('/api/v1/jobs', methods=['POST'])
def booking():
    payload = request.get_json()
    inserted_id = mongo.db.jobs.insert_one(payload).inserted_id
    return jsonify({"message": "Job created", "job_id": str(inserted_id)}), 201

@app.route('/api/v1/workshops/<int:workshop_id>/jobs', methods=['GET'])
def workshop_advisor(workshop_id):
    cursor = mongo.db.jobs.find({"workshop_id": workshop_id})
    return dumps(list(cursor)), 200

@app.route('/api/v1/workshops/jobs', methods=['GET'])
def worshop():
    data=list(mongo.db.jobs.find({}))
    return data,200
if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
