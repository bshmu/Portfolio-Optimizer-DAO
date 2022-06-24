from flask import Flask
from scripts.engine import engine

api = Flask(__name__)

@api.route('/optimizeFundWeights')

def optimizeFundWeights():
    engine()  # call the engine
