import os
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from web3 import Web3, exceptions
import utils.contract_utils as cu
from utils.portfolio_optimizer_utils import get_intraday_crypto_time_series

def get_asset_holdings(asset, tokens, holdings, dtime, interval='5min', output_size='compact'):
    """
    Gets the holdings for the given asset for the relevant proposal in USD along with a long or short tag.
    """
    # Loop through the lists to get the holdings in base token terms
    for i in range(len(tokens)):
        if tokens[i] == asset:
            if holdings[i] != 0:
                holding = holdings[i] / 10**18
                ls = 'long'
            elif holdings[i] == 0 and holdings[i + len(cu.contract_tokens)] != 0:
                holding = holdings[i + len(cu.contract_tokens)] / 10**18
                ls = 'short'
            else:
                holding = 0
                ls = ''

    # Convert to USD using the dtime
    mapped_asset = asset if asset not in list(cu.contract_tokens_mapping.keys()) else cu.contract_tokens_mapping[asset]
    token_price_series = get_intraday_crypto_time_series(mapped_asset, start_time=dtime, interval=interval, output_size=output_size)
    token_price = token_price_series.iloc[0]
    holding_usd = holding * token_price

    return (holding_usd, ls)

def get_cumulative_performance():
    # Load dotenv
    load_dotenv()

    # Connect to infura
    w3 = Web3(Web3.HTTPProvider("https://rinkeby.infura.io/v3/" + os.getenv('WEB3_INFURA_PROJECT_ID')))

    # Get the contract object
    deployed_contract_address = cu.contract_address
    abi = cu.get_contract_abi()
    dao = w3.eth.contract(address=deployed_contract_address, abi=abi)

    # Get the number of proposals
    performance_history = [0.0]
    num_proposals = dao.functions.lengthOfProposals().call()

    for i in range(num_proposals):
        # Try catch block because lengthOfProposals has some issues
        try:
            proposal_i = dao.functions.getHoldingsDataOfProposal(i).call()
        except exceptions.ContractLogicError:
            print("Error on proposal " + str(i + 1) + ', breaking...')
            break

        # Get the data from proposal i
        tokens = proposal_i[0]
        holdings = proposal_i[1]
        timestamps = proposal_i[3]

        # For debugging, make sure end time is not zero
        if timestamps[1] == 0:
            timestamps[1] = timestamps[0] + 300
        else:
            pass

        # Convert the timestamps
        timestamps = [datetime.utcfromtimestamp(int(ts)) for ts in timestamps]
        start_time = timestamps[0]
        end_time = timestamps[1]

        # Convert the holdings into weights using the pricing API and get into a dictionary
        asset_holdings_dict = dict.fromkeys(cu.contract_tokens)
        for asset in list(asset_holdings_dict.keys()):
            asset_holdings, asset_ls = get_asset_holdings(asset, tokens, holdings, start_time)
            asset_holdings_dict[asset] = (asset_holdings, asset_ls)

        # Convert the holdings to weights using the token prices as of the starting timestamp
        total_usd_holdings = 0
        for asset in list(asset_holdings_dict.keys()):
            total_usd_holdings += asset_holdings_dict[asset][0]
        asset_weights_dict = dict.fromkeys(cu.contract_tokens)
        for asset in list(asset_weights_dict.keys()):
            asset_weights_dict[asset] = (asset_holdings_dict[asset][0] / total_usd_holdings, asset_holdings_dict[asset][1])

        # Get the performance for each token using the weights
        asset_performance_dict = dict.fromkeys(cu.contract_tokens)
        for asset in list(asset_performance_dict.keys()):
            mapped_asset = asset if asset not in list(cu.contract_tokens_mapping.keys()) else cu.contract_tokens_mapping[asset]
            asset_price_series = get_intraday_crypto_time_series(mapped_asset, start_time=start_time, end_time=end_time)
            start_price = asset_price_series.iloc[0]
            end_price = asset_price_series.iloc[-1]
            weight = asset_weights_dict[asset][0]
            ls = asset_weights_dict[asset][1]
            if ls == 'long':
                asset_performance_dict[asset] = weight * (end_price / start_price - 1)
            elif ls == 'short':
                asset_performance_dict[asset] = weight * (end_price / start_price - 1) * -1
            else:
                asset_performance_dict[asset] = 0

        # Calculate the period return by summing the asset returns and append to the performance history list
        proposal_fund_performance = 0
        for asset in list(asset_performance_dict.keys()):
            proposal_fund_performance += asset_performance_dict[asset]
        performance_history.append(proposal_fund_performance)

    # Get a cumulative performance
    cumulative_performance_history = ((1 + pd.Series(performance_history)).cumprod()).tolist()
    cumulative_performance_history = [c * 100 for c in cumulative_performance_history]

    return cumulative_performance_history

if __name__ == '__main__':
    c = get_cumulative_performance()