from flask import Flask
from scripts.engine import engine, initiateTradesOnUniswap
from scripts.performance import get_cumulative_performance

api = Flask(__name__)

@api.route('/optimizeFundWeights', methods=["POST"])

def optimizeFundWeights():
    assets, percentages = engine()  # call the engine
    initiateTradesOnUniswap(assets, percentages)  # initiate trades

@api.route('/getCumulativePerformance')

def getCumulativePerformance():
    return get_cumulative_performance()

if __name__ == "__main__":
    api.run("localhost", 6000)
