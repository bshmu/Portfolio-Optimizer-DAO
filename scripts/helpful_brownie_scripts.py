import os
import json
from dotenv import load_dotenv
from solcx import compile_standard, install_solc
from web3 import Web3
from web3.middleware import geth_poa_middleware
from brownie import OptimizerDAO, accounts, config

"""
1) Deploying a contract using Brownie

- comment out all code except the deployment part
- brownie compile
- brownie run scripts/helpful_brownie_scripts.py --network rinkeby

2) Getting contract ABI

- brownie console
- >> from brownie import myContract
- >> import json
- >> abi = myContract.abi
- >> json_string = json.dumps(abi)
- >> with open(r"~/filepath.../test_abi.json", "w") as f: 
        json.dump(json_string, f)
"""

# Get the private key from the config file
account = accounts.add(config["wallets"]["from_key"])

# Deploy the contract
def deploy():
    optimizerDAO = OptimizerDAO.deploy({"from": account,  "priority_fee": 35000000000}, publish_source=config['networks']['rinkeby']['verify'])
    print(f'Successfully deployed OptimizerDAO contract to: {optimizerDAO.address}')

deploy()

# ABI and address
abi_path = r"C:\Users\User\repos\chainshotdev\utils\test_abi.json"
with open(abi_path, 'r') as j:
    abi = json.loads(j.read())
deployed_address = "0xd5a2DcdE322549AC9A71a55e1686ec4B8B739583"

# Connect to infura
load_dotenv()
w3 = Web3(Web3.HTTPProvider("https://rinkeby.infura.io/v3/" + os.getenv('WEB3_INFURA_PROJECT_ID')))

# Contract
optimizerDAO = w3.eth.contract(address=deployed_address, abi=abi)