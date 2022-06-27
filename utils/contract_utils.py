import os
import json

# Addresses
metamask_address = '0x52299706E6aC1d97e3d8d4526Fa8554903e16a1d'  # Update to your metamask address

contract_address = '0x5411Af80B34484F2ded92998205AAB4f68599959'

contract_tokens = ['WBTC', 'WETH', 'UNI', 'BAT', 'USDT']

contract_tokens_mapping = {'WBTC': 'BTC', 'WETH': 'ETH'}

vote_fields = ['tokens', 'view', 'confidence', 'viewType', 'viewRelativeToken']

def get_contract_abi():
    json_path = os.path.dirname(__file__) + '/compiled_sol.json'
    with open(json_path, 'r') as file:
        output_file = file.read()
    abi = json.loads(output_file)
    return abi