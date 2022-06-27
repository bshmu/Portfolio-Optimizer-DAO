from flask import Flask
from scripts.engine import engine
from scripts.performance import get_cumulative_performance

api = Flask(__name__)

@api.route('/optimizeFundWeights')

def optimizeFundWeights():
    assets, percentages = engine()  # call the engine

@api.route('/getCumulativePerformance')

def getCumulativePerformance():
    return get_cumulative_performance()  # [1, 1.05, 0.99, ...]