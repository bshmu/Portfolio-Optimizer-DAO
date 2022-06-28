Back-end Dependencies (Python):
 - Numpy
 - Pandas
 - Scipy
 - Flask
 - dotenv
 - Web3 (https://web3py.readthedocs.io/en/stable/quickstart.html)
 - Brownie (https://eth-brownie.readthedocs.io/en/latest/index.html)

API Dependencies:
 - AlphaVantage: off-chain historical token prices and market caps (www.alphavantage.co)
 - Infura: Ethereum blockchain development platform compatible with Brownie (www.infura.io)

Brownie Installation Instructions:
https://codeburst.io/deploy-a-smart-contract-using-python-how-to-b62de0124b

Black-Litterman Model Resources:
 - PyPortfolioOpt Repo: https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html
 - "The Black-Litterman Model in Detail": https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1314585


Front-end Dependencies:
- NextJs
- Npm 
- Flask
- Axios
- Tailwind
- Web3modal
- WalletConnect

Starting NextJs: 
cd portfolio-frontend
run npm install axios
run npm install -D tailwindcss postcss autoprefixer
run npx tailwindcss init -p
run npm install --save web3modal
run npm install --save ethers
run npm install --save @walletconnect/web3-provider
run npm install react-toastify
run npm run dev 

Installing Flask: 
cd portfolio-frontend/flask-backend
run python -m venv env 
run source env/bin/activate
run pip install flask
run pip install python-dotenv

Starting Flask Backend: 
cd portfolio-frontend/flask-backend
run dev-start-flask

Navigate to the url http://127.0.0.1:5000/profile for preview of the response_body rendered in JSON format
