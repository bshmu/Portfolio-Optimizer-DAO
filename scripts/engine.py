import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from web3 import Web3
import utils.contract_utils as cu
from optimizer import BlackLittermanOptimizer

def parse_views(votes):
    """
    Parses views/confidences from the DAO into data that can be entered into the optimizer.
    """
    # Get the votes into a dataframe
    votes_to_dict = {fld: votes for fld, votes in zip(cu.vote_fields, votes)}
    votes_df = pd.DataFrame(votes_to_dict)
    votes_df['view'] /= 100.0
    votes_df['confidence'] /= 100.0

    # First groupby sum
    votes_df_by_view_type = votes_df.groupby(['viewType'])['tokens'].sum()
    tokens_absolute = votes_df_by_view_type.loc['absolute'] if 'absolute' in votes_df_by_view_type.index else 0
    tokens_relative = votes_df_by_view_type.loc['relative'] if 'relative' in votes_df_by_view_type.index else 0

    # If more tokens made an absolute type vote, return the weighted average view/confidence
    if tokens_absolute >= tokens_relative:
        subset = votes_df.loc[votes_df['viewType'] == 'absolute']
        view = float(np.average(subset['view'].to_numpy(), weights=subset['tokens'].to_numpy()))
        confidence = float(np.average(subset['confidence'].to_numpy(), weights=subset['tokens'].to_numpy()))
        relative_token = ''

    # If more tokens made a relative view type, return the weighted average view/confidence for the highest voted relative token
    else:
        subset = votes_df.loc[votes_df['viewType'] == 'relative']
        subset_agg = subset.groupby('viewRelativeToken')['tokens'].sum().reset_index()

        # Choose most voted on token in the subset
        relative_token = subset_agg[subset_agg['tokens'] == subset_agg['tokens'].max()]['viewRelativeToken'].iloc[0]
        subset_relative_token = subset.loc[subset['viewRelativeToken'] == relative_token]
        view = float(np.average(subset_relative_token['view'].to_numpy(),
                                weights=subset_relative_token['tokens'].to_numpy()))
        confidence = float(np.average(subset_relative_token['confidence'].to_numpy(),
                                      weights=subset_relative_token['tokens'].to_numpy()))

    # Final sanity check
    assert(type(view).__name__ == 'float')
    assert(type(confidence).__name__ == 'float')
    assert(type(relative_token).__name__ == 'str')

    return view, confidence, relative_token

def parse_optimal_weights(optimal_weights):
    """
    Parses optimal weights from the optimizer for pass-through to initiate trades on Uniswap.
    """
    assets = list(optimal_weights.keys())
    full_assets = assets + ['S' + asset for asset in assets]
    ref = dict.fromkeys(full_assets, None)
    for asset in assets:
        if optimal_weights[asset][1] == 'long':
            ref[asset] = int(np.round(optimal_weights[asset][0] * 100))
            ref['S' + asset] = 0
        else:
            ref['S' + asset] = int(np.round(optimal_weights[asset][0] * 100))
            ref[asset] = 0
    full_percentages = list(ref.values())

    return full_assets, full_percentages

def engine():
    """
    This function makes a connection to the Infura API, loads the DAO contract using the Web3 library, parses the views
    and confidences stored in the contract's proposals into data that can enter the optimizer, calculates the optimal
    weights, and feeds them back into the contract's "initiateTradesOnUniswap" function.
    """

    # Load dotenv
    load_dotenv()

    # Connect to infura
    w3 = Web3(Web3.HTTPProvider("https://rinkeby.infura.io/v3/" + os.getenv('WEB3_INFURA_PROJECT_ID')))

    # Get the contract object
    deployed_contract_address = cu.contract_address
    json_path = os.path.dirname(__file__) + '/compiled_sol.json'
    with open(json_path, 'r') as file:
        output_file = file.read()
    abi = json.loads(output_file)["abi"]
    dao = w3.eth.contract(address=deployed_contract_address, abi=abi)

    # For each token, parse the views and confidences
    views = {}
    views_confidences = {}
    for token in cu.contract_tokens:
        print("Parsing views for", token, '...')

        # Get the votes from the DAO
        token_votes = dao.functions.getProposalVotes(token).call()
        if len(token_votes[0]) == 0:
            continue

        # Parse views
        token_view, token_confidence, token_relative_token = parse_views(token_votes)

        # Append to the dictionaries
        if token in list(cu.contract_tokens_mapping.keys()):
            token = cu.contract_tokens_mapping[token]

        views[token] = (token_view, token_relative_token)
        views_confidences[token] = token_confidence

    # Call the optimizer
    optimizer = BlackLittermanOptimizer(cu.contract_tokens, views, views_confidences)
    weights = optimizer.normalizedWeights

    # Read the optimal weights back into values that can enter the contract to execute Uniswap trades
    assets, percentages = parse_optimal_weights(weights)

    # Create transaction to call initiateTradesOnUniswap -- the transaction may take several minutes to clear
    nonce = w3.eth.get_transaction_count(cu.metamask_address) + 1
    transaction_dict = {'gas': 70000,
                        'gasPrice': w3.toWei('21', 'gwei'),
                        'from': cu.metamask_address,
                        'nonce': nonce}
    txn = dao.functions.initiateTradesOnUniswap(assets, percentages).buildTransaction(transaction_dict)
    signed_txn = w3.eth.account.signTransaction(txn, private_key=os.getenv('PRIVATE_KEY'))
    w3.eth.sendRawTransaction(signed_txn.rawTransaction)

    return 0

