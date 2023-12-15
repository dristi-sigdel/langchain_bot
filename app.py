from flask import Flask, request, jsonify
from model import load_llm

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/parse', methods=['POST', 'GET'])
def process_json():
    try:
        # Get JSON data from the request
        input_data = request.get_json()["value"]
        print(input_data)

        # Perform some processing on the input data (you can replace this with your own logic)
        prediction = load_llm(input_data)
        processed_data = {"output": prediction}

        # Return the processed data as JSON
        return jsonify(processed_data)

    except Exception as e:
        # Handle any exceptions that may occur during processing
        error_message = {"error": str(e)}
        return jsonify(error_message), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,  debug=True)
