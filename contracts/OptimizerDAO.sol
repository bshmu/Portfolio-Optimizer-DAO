//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.0;

import "hardhat/console.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";



interface IERC20Master {
    function totalSupply() external view returns (uint);
    function balanceOf(address account) external view returns (uint);
    function transfer(address recipient, uint amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint);
    function approve(address spender, uint amount) external returns (bool);
    function transferFrom(
        address sender,
        address recipient,
        uint amount
    ) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint value);
    event Approval(address indexed owner, address indexed spender, uint value);
}


//import the uniswap router
//the contract needs to use swapExactTokensForTokens
//this will allow us to import swapExactTokensForTokens into our contract

interface IUniswapV2Router {
  function getAmountsOut(uint256 amountIn, address[] memory path)
    external
    view
    returns (uint256[] memory amounts);

  function swapExactTokensForTokens(

    //amount of tokens we are sending in
    uint256 amountIn,
    //the minimum amount of tokens we want out of the trade
    uint256 amountOutMin,
    //list of token addresses we are going to trade in.  this is necessary to calculate amounts
    address[] calldata path,
    //this is the address we are going to send the output tokens to
    address to,
    //the last time that the trade is valid for
    uint256 deadline
  ) external returns (uint256[] memory amounts);
}

interface IUniswapV2Pair {
  function token0() external view returns (address);
  function token1() external view returns (address);
  function swap(
    uint256 amount0Out,
    uint256 amount1Out,
    address to,
    bytes calldata data
  ) external;
}

interface IUniswapV2Factory {
  function getPair(address token0, address token1) external returns (address);
}



contract OptimizerDAO is ERC20 {
  // May be able to delete membersTokenCount as tally is taken care of in ERC contract
  uint public treasuryEth;
  uint public startingEth;
  uint public lastSnapshotEth;

  address private constant UNISWAP_V2_ROUTER = 0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f;
  address private constant WETH = 0xc778417E063141139Fce010982780140Aa0cD5Ab;

  mapping(string => address) private tokenAddresses;
  // Address's included in mapping
  /**
  address private constant WETH = 0xc778417E063141139Fce010982780140Aa0cD5Ab;
  address private constant BAT = 0xDA5B056Cfb861282B4b59d29c9B395bcC238D29B;
  address private constant WBTC = 0x0014F450B8Ae7708593F4A46F8fa6E5D50620F96;
  address private constant UNI = 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984;
  address private constant MKR = 0xF9bA5210F91D0474bd1e1DcDAeC4C58E359AaD85;
  */

  //
  mapping(string => uint) public assetWeightings;

  // Proposal struct of token, expected performance and confidence level.
  struct Proposal {
    uint date;
    uint endDate;
    string[] tokens;
    // Maps Token (i.e 'btc') to array
    mapping(string => uint[]) numOfUserTokens;
    // Maps Token string to array of total token amount
    mapping(string => uint[]) userViews;
    mapping(string => uint[]) userConfidenceLevel;
  }

  // Array of Proposals
  Proposal[] public proposals;


  constructor() ERC20("Optimizer DAO Token", "ODP") {
    // On DAO creation, a vote/proposal is created which automatically creates a new one every x amount of time
    Proposal storage proposal = proposals.push();
    proposal.date = block.timestamp;
    string[5] memory _tokens = ["WETH", "BAT", "WBTC", "UNI", "MKR"];
    address[5] memory _addresses = [0xc778417E063141139Fce010982780140Aa0cD5Ab, 0xDA5B056Cfb861282B4b59d29c9B395bcC238D29B, 0x0014F450B8Ae7708593F4A46F8fa6E5D50620F96, 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984, 0xF9bA5210F91D0474bd1e1DcDAeC4C58E359AaD85];
    for (uint i = 0; i < _tokens.length; i++) {
      tokenAddresses[_tokens[i]] = _addresses[i];
    }
  }



  function joinDAO() public payable {
    // What is the minimum buy in for the DAO?
    require(msg.value >= 1 ether, "Minimum buy in is 1 ether");

    if (treasuryEth == 0) {

      // If there is nothing in the treasury, provide liquidity to treasury
      // LP tokens are initially provided on a 1:1 basis
      treasuryEth = msg.value;
      startingEth = treasuryEth;
      console.log(treasuryEth);
      // change to _mint
      _mint(msg.sender, treasuryEth);

    } else {
      // DAO members token count is diluted as more members join / add Eth
      treasuryEth += msg.value;
      startingEth = treasuryEth;
      uint ethReserve =  treasuryEth - msg.value;
      uint proportionOfTokens = (msg.value * totalSupply()) / ethReserve;
      // change to _mint
      _mint(msg.sender, proportionOfTokens);
    }
  }


  function leaveDAO() public {
    uint tokenBalance = balanceOf(msg.sender);
    require(tokenBalance > 0);

    // User gets back the relative % of the
    uint ethToWithdraw = (tokenBalance / totalSupply()) * treasuryEth;
    _burn(msg.sender, tokenBalance);
    payable(msg.sender).transfer(ethToWithdraw);
    treasuryEth -= ethToWithdraw;
  }


  function submbitVote(string[] memory _token, uint[] memory _perfOfToken, uint[] memory _confidenceLevels) public onlyMember {
    // User inputs token they'd like to vote on, the expected performance of token over time period and their confidence level
    // Loop through each token in list and provide a +1 on list
    // If token is in proposal, include in Struct and output average for Performance & confidence levels
    require((_token.length == _perfOfToken.length) && (_perfOfToken.length == _confidenceLevels.length), "Arrays must be the same size");

    uint numberOfVoterTokens = balanceOf(msg.sender);
    for (uint i = 0; i < _token.length; i++) {
      // get each value out of array
      proposals[proposals.length - 1].tokens.push(_token[i]);
      proposals[proposals.length - 1].userViews[_token[i]].push(_perfOfToken[i]);

      proposals[proposals.length - 1].numOfUserTokens[_token[i]].push(numberOfVoterTokens);
      proposals[proposals.length - 1].userConfidenceLevel[_token[i]].push(_confidenceLevels[i]);

    }
  }

  function getProposalVotes(string memory _token) public view returns (uint[] memory, uint[] memory, uint[] memory){
      Proposal storage proposal = proposals[proposals.length - 1];
      uint length = proposal.numOfUserTokens[_token].length;
      uint[]    memory _numOfUserTokens = new uint[](length);
      uint[]  memory _userViews = new uint[](length);
      uint[]    memory _userConfidenceLevel = new uint[](length);

      for (uint i = 0; i < length; i++) {
          console.log(proposal.numOfUserTokens[_token][i]);
          _numOfUserTokens[i] = proposal.numOfUserTokens[_token][i];
          _userViews[i] = proposal.userViews[_token][i];
          _userConfidenceLevel[i] = proposal.userConfidenceLevel[_token][i];
      }

      return (_numOfUserTokens, _userViews, _userConfidenceLevel);

  }

  // Event to emit for Python script to pick up data for model?


  /**
  function findTokenWeight() public  {
    uint sumOfLPForToken;

    uint numeratorToken;
    uint numeratorConfidence;

    for (uint i = 0; i < proposals[proposals.length - 1].tokens.length; i++) {
      string memory _token = proposals[proposals.length - 1].tokens[i];
      sumOfLPForToken += proposals[proposals.length - 1].numOfUserTokens[_token][i];
      numeratorToken += proposals[proposals.length - 1].numOfUserTokens[_token][i] * proposals[proposals.length - 1].userWeightings[_token][i];
      numeratorConfidence += proposals[proposals.length - 1].numOfUserTokens[_token][i] * proposals[proposals.length - 1].userConfidenceLevel[_token][i];

      uint weightedAveragePerformance = numeratorToken / sumOfLPForToken;
      uint weightedAverageConfidence = numeratorConfidence / sumOfLPForToken;

      // This will return a number with 18 decimals, need to divide by 18
      proposals[proposals.length - 1].proposalFinalPerformance[_token] = weightedAveragePerformance;
      proposals[proposals.length - 1].proposalFinalConfidence[_token] = weightedAverageConfidence;
    }


    // Update Token weightings mapping
    // initialize tradesOnUniswap function

  }
  */

  function initiateTradesOnUniswap(string[] memory _assets, uint[] memory _percentage) public {

    if (proposals.length > 0) {
      // 1. Sell off existing holdings
      for (uint i = 0; i < _assets.length; i++) {
        // Asset swapping from, to WETH, transfer whole balance, recipient is the SC
        _swap(tokenAddresses[_assets[i]], tokenAddresses["WETH"], ERC20(tokenAddresses[_assets[i]]).balanceOf(address(this)), 0, address(this));
      }
      // 2. Take a snapshot of the proceedings in WETH
      lastSnapshotEth = ERC20(tokenAddresses["WETH"]).balanceOf(address(this));

      // 3. Convert any Eth in treasury to WETH
      (bool success, ) = WETH.call{value: address(this).balance}(abi.encodeWithSignature("deposit()"));
      require(success, "The transaction failed");

      // 4. Reallocate all WETH based on new weightings
      for (uint i = 0; i < _assets.length; i++) {
        uint allocation = _percentage[i] * ERC20(tokenAddresses["WETH"]).balanceOf(address(this));
        _swap(WETH, tokenAddresses[_assets[i]], allocation, 0, address(this));
      }

      // 4. Create new proposal
      Proposal storage newProposal = proposals.push();
      newProposal.date = block.timestamp;

    } else {
      // 1. If first Proposal, convert all Eth to WETH
      (bool success, ) = WETH.call{value: address(this).balance}(abi.encodeWithSignature("deposit()"));
      require(success, "The transaction failed");

      // 2. Take asset weightings and purchase assets
      for (uint i = 0; i < _assets.length; i++) {
        uint allocation = _percentage[i] * ERC20(tokenAddresses["WETH"]).balanceOf(address(this));
        _swap(WETH, tokenAddresses[_assets[i]], allocation, 0, address(this));
      }

      // 3. Create new proposal
      Proposal storage newProposal = proposals.push();
      newProposal.date = block.timestamp;

    }

  }

  //this swap function is used to trade from one token to another
    //the inputs are self explainatory
    //token in = the token address you want to trade out of
    //token out = the token address you want as the output of this trade
    //amount in = the amount of tokens you are sending in
    //amount out Min = the minimum amount of tokens you want out of the trade
    //to = the address you want the tokens to be sent to

  function _swap(address _tokenIn, address _tokenOut, uint256 _amountIn, uint256 _amountOutMin, address _to) public {

    //first we need to transfer the amount in tokens from the msg.sender to this contract
    //this contract will have the amount of in tokens
    //IERC20Master(_tokenIn).transferFrom(msg.sender, address(this), _amountIn);

    //next we need to allow the uniswapv2 router to spend the token we just sent to this contract
    //by calling IERC20 approve you allow the uniswap contract to spend the tokens in this contract
    ERC20(_tokenIn).approve(UNISWAP_V2_ROUTER, _amountIn);

    //path is an array of addresses.
    //this path array will have 3 addresses [tokenIn, WETH, tokenOut]
    //the if statement below takes into account if token in or token out is WETH.  then the path is only 2 addresses
    address[] memory path;
    if (_tokenIn == WETH || _tokenOut == WETH) {
      path = new address[](2);
      path[0] = _tokenIn;
      path[1] = _tokenOut;
    } else {
      path = new address[](3);
      path[0] = _tokenIn;
      path[1] = WETH;
      path[2] = _tokenOut;
    }
        //then we will call swapExactTokensForTokens
        //for the deadline we will pass in block.timestamp
        //the deadline is the latest time the trade is valid for
        IUniswapV2Router(UNISWAP_V2_ROUTER).swapExactTokensForTokens(_amountIn, _amountOutMin, path, _to, block.timestamp);
    }


  modifier onlyMember {
      require(balanceOf(msg.sender) > 0);
      _;
   }

}
