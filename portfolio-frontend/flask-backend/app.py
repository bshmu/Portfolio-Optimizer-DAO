from flask import Flask
from scripts.engine import engine

api = Flask(__name__)

@api.route('/optimizeFundWeights', methods=["POST"])

def optimizeFundWeights():
    engine()  # call the engine

if __name__ == "__main__":
    api.run("localhost", 6000)
