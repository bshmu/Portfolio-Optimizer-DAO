import os
import json

# Addresses
metamask_address = '0x52299706E6aC1d97e3d8d4526Fa8554903e16a1d'  # Update to your metamask address

contract_address = '0x9e7Ddca6b3646711880651652eF58667469E51a3'  # Update to deployed contract address

contract_tokens = ['WETH', 'BAT', 'WBTC', 'UNI', 'USDT']  # Order matters for initiateTradesOnUniswap()

contract_tokens_mapping = {'WBTC': 'BTC', 'WETH': 'ETH'}

vote_fields = ['tokens', 'view', 'confidence', 'viewType', 'viewRelativeToken']

def get_contract_abi():
    json_path = os.path.dirname(__file__) + '/abi.json'
    with open(json_path, 'r') as file:
        output_file = file.read()
    abi = json.loads(output_file)
    return abi