//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.0;

import "hardhat/console.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

// interface of ERC20 movements/events
interface IERC20Master {
    // total supply of token
    function totalSupply() external view returns (uint);

    // balance of individual addresses of an account
    function balanceOf(address account) external view returns (uint);

    // transfer token from one address to another
    function transfer(address recipient, uint amount) external returns (bool);

    // allowance of token to be spent by another address
    function allowance(
        address owner, // does the owner assign the spender?
        address spender
    ) external view returns (uint);

    // approves the spender to spend the owners tokens
    function approve(address spender, uint amount) external returns (bool);

    // idk why there are two transfer functions
    function transferFrom(
        address sender,
        address recipient,
        uint amount
    ) external returns (bool);

    event Approval(address indexed owner, address indexed spender, uint value);
    event Transfer(address indexed from, address indexed to, uint value);
}

//import the uniswap router
//the contract needs to use swapExactTokensForTokens
//this will allow us to import swapExactTokensForTokens into our contract

interface WETH9 {
    function balanceOf(address _address) external returns (uint256);

    function deposit() external payable;
}

interface ERC20short {
    function mint(address _address, uint _amount) external;

    function burn(address _address, uint _amount) external;

    function balanceOf(address _account) external view returns (uint256);
}

interface IUniswapV2Router {
    function getAmountsOut(
        uint256 amountIn,
        address[] memory path
    ) external view returns (uint256[] memory amounts);

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

contract OptimizerDAO is ERC20, Ownable {
    uint256 currentEthReserve;
    uint256 lastSnapshotEth;
    address private constant UNISWAP_V2_ROUTER =
        0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;
    address private constant WETH = 0xc778417E063141139Fce010982780140Aa0cD5Ab;
    ISwapRouter router =
        ISwapRouter(0xE592427A0AEce92De3Edee1F18E0157C05861564);

    mapping(string => address) private tokenAddresses;
    mapping(string => address) private shortTokenAddresses;

    //
    mapping(string => uint) public assetWeightings;

    // Proposal struct of token, expected performance and confidence level.
    struct Proposal {
        uint startTime;
        uint endTime;
        uint startEth;
        uint endEth;
        string[] tokens;
        uint[] qtyOfTokensAcq;
        uint[] tokensWeightings;
        mapping(string => uint[]) numOfUserTokens;
        mapping(string => uint[]) userViews;
        mapping(string => uint[]) userConfidenceLevel;
        mapping(string => string[]) userViewsType;
        mapping(string => string[]) userViewsRelativeToken;
    }

    // Array of Proposals
    Proposal[] public proposals;

    constructor() ERC20("Optimizer DAO Token", "ODP") {
        // On DAO creation, a vote/proposal is created which automatically creates a new one every x amount of time
        Proposal storage newProposal = proposals.push();
        string[2] memory _tokens = ["ETH", "UNI"];
        address[2] memory _addresses = [
            0xB4FBF271143F4FBf7B91A5ded31805e42b2208d6,
            0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984
        ];

        // string[5] memory _shortTokens = [
        //     "sWETH",
        //     "sBAT",
        //     "sWBTC",
        //     "sUNI",
        //     "sUSDT"
        // ];
        // address[5] memory _shortAddresses = [
        //     0x982cd41387dd65e659279C4EFCF05c25c4B586D6,
        //     0xf760954e01e53c3f7F08733ca1dC62B14b4BF50e,
        //     0xA18533Ba93407a54BB1bcaDB7e9f3D34e46039F9,
        //     0x95552cA5cc9f329E5376659eaD39F880307B7A13,
        //     0x1c0b9527210B427ad9bdfF41bb3a3b78C9ceE7d9
        // ];

        // initializes arrays with proper short/long addresses
        for (uint i = 0; i < _tokens.length; i++) {
            tokenAddresses[_tokens[i]] = _addresses[i];
            // shortTokenAddresses[_shortTokens[i]] = _shortAddresses[i];
        }
    }

    function joinDAO() public payable {
        // Minimum buy in at 0.1 Eth
        require(msg.value >= 0.1 ether, "Minimum buy in is 0.1 ether");

        uint256 initialBalance = address(this).balance - msg.value;

        if (initialBalance == 0) {
            // If there is nothing in the treasury, provide liquidity to treasury
            // LP tokens are initially provided on a 1:1 basis
            currentEthReserve = msg.value;
            _mint(msg.sender, currentEthReserve * 10 ** 18);
        } else {
            // DAO members token count is diluted as more members join / add Eth
            initialBalance += msg.value;

            uint previousEthBalance = initialBalance - msg.value;
            uint proportionOfTokens = (msg.value * totalSupply()) /
                previousEthBalance;
            _mint(msg.sender, proportionOfTokens * 10 ** 18);
        }
    }

    // function leaveDAO() public {
    //     uint tokenBalance = balanceOf(msg.sender);
    //     require(tokenBalance > 0);

    //     // User gets back the relative % of the
    //     uint ethToWithdraw = (tokenBalance / totalSupply()) * treasuryEth;
    //     _burn(msg.sender, tokenBalance);
    //     payable(msg.sender).transfer(ethToWithdraw);
    //     treasuryEth -= ethToWithdraw;
    // }

    // this function allows the user to choose token of their choice and their respective actions
    function submitVote(
        string[] memory _token,
        uint[] memory _prefOfToken,
        uint[] memory _confidenceLevels,
        string[] memory _userViewsType,
        string[] memory _userViewsRelativeToken
    ) public onlyMember {
        // User inputs token they'd like to vote on, the expected performance of token over time period and their confidence level
        // Loop through each token in list and provide a +1 on list
        // If token is in proposal, include in Struct and output average for Performance & confidence levels
        require(
            (_token.length == _prefOfToken.length) &&
                (_prefOfToken.length == _confidenceLevels.length),
            "Arrays must be the same size"
        );

        uint numberOfVoterTokens = balanceOf(msg.sender);
        for (uint i = 0; i < _token.length; i++) {
            proposals[proposals.length - 1].userViews[_token[i]].push(
                _prefOfToken[i]
            );

            proposals[proposals.length - 1].numOfUserTokens[_token[i]].push(
                numberOfVoterTokens
            );
            proposals[proposals.length - 1].userConfidenceLevel[_token[i]].push(
                _confidenceLevels[i]
            );
            proposals[proposals.length - 1].userViewsType[_token[i]].push(
                _userViewsType[i]
            );
            proposals[proposals.length - 1]
                .userViewsRelativeToken[_token[i]]
                .push(_userViewsRelativeToken[i]);
        }
    }

    function getProposalVotes(
        string memory _token
    )
        public
        view
        returns (
            uint[] memory,
            uint[] memory,
            uint[] memory,
            string[] memory,
            string[] memory
        )
    {
        Proposal storage proposal = proposals[proposals.length - 1];
        uint length = proposal.numOfUserTokens[_token].length;
        uint[] memory _numOfUserTokens = new uint[](length);
        uint[] memory _userViews = new uint[](length);
        uint[] memory _userConfidenceLevel = new uint[](length);
        string[] memory _userViewsType = new string[](length);
        string[] memory _userViewsRelativeToken = new string[](length);

        for (uint i = 0; i < length; i++) {
            console.log(proposal.numOfUserTokens[_token][i]);
            _numOfUserTokens[i] = proposal.numOfUserTokens[_token][i];
            _userViews[i] = proposal.userViews[_token][i];
            _userConfidenceLevel[i] = proposal.userConfidenceLevel[_token][i];
            _userViewsType[i] = proposal.userViewsType[_token][i];
            _userViewsRelativeToken[i] = proposal.userViewsRelativeToken[
                _token
            ][i];
        }

        return (
            _numOfUserTokens,
            _userViews,
            _userConfidenceLevel,
            _userViewsType,
            _userViewsRelativeToken
        );
    }

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

    function initiateTradesOnUniswap(
        string[] memory _assets,
        uint[] memory _percentage
    ) public {
        bytes32 wethRepresentation = keccak256(abi.encodePacked("WETH"));

        if (proposals.length > 1) {
            // 1. Sell off existing holdings

            for (uint i = 0; i < _assets.length; i++) {
                if (tokenAddresses[_assets[i]] != address(0)) {
                    if (
                        ERC20(tokenAddresses[_assets[i]]).balanceOf(
                            address(this)
                        ) >
                        0 &&
                        (keccak256(abi.encodePacked(_assets[i])) !=
                            wethRepresentation)
                    ) {
                        _swap(
                            tokenAddresses[_assets[i]],
                            WETH,
                            ERC20(tokenAddresses[_assets[i]]).balanceOf(
                                address(this)
                            ),
                            0,
                            address(this)
                        );
                    }
                } else if (
                    shortTokenAddresses[_assets[i]] != address(0) &&
                    ERC20short(shortTokenAddresses[_assets[i]]).balanceOf(
                        address(this)
                    ) >
                    0
                ) {
                    ERC20short(shortTokenAddresses[_assets[i]]).burn(
                        address(this),
                        ERC20short(shortTokenAddresses[_assets[i]]).balanceOf(
                            address(this)
                        )
                    );
                }
            }
            // 2. Take a snapshot of the proceedings in WETH & time for each proposal
            proposals[proposals.length - 2].endEth = WETH9(WETH).balanceOf(
                address(this)
            );
            proposals[proposals.length - 2].endTime = block.timestamp;
            proposals[proposals.length - 1].startEth = WETH9(WETH).balanceOf(
                address(this)
            );
            proposals[proposals.length - 1].startTime = block.timestamp;

            lastSnapshotEth = WETH9(WETH).balanceOf(address(this));

            // 3. Convert any Eth in contract from new members to WETH
            WETH9(WETH).deposit{value: address(this).balance}();

            // 5. Reallocate all WETH based on new weightings
            for (uint i = 0; i < _assets.length; i++) {
                assetWeightings[_assets[i]] = _percentage[i];
                proposals[proposals.length - 1].tokensWeightings.push(
                    _percentage[i]
                );
                proposals[proposals.length - 1].tokens.push(_assets[i]);

                if (_percentage[i] == 0) {
                    proposals[proposals.length - 1].qtyOfTokensAcq.push(0);
                } else {
                    if (tokenAddresses[_assets[i]] == WETH) {
                        uint allocation = (lastSnapshotEth * _percentage[i]) /
                            100;
                        proposals[proposals.length - 1].qtyOfTokensAcq.push(
                            allocation
                        );
                    }
                    if (
                        _percentage[i] != 0 &&
                        (keccak256(abi.encodePacked(_assets[i])) !=
                            wethRepresentation)
                    ) {
                        if (
                            tokenAddresses[_assets[i]] != address(0) &&
                            _percentage[i] != 0
                        ) {
                            uint allocation = (lastSnapshotEth *
                                _percentage[i]) / 100;
                            _swap(
                                WETH,
                                tokenAddresses[_assets[i]],
                                allocation,
                                0,
                                address(this)
                            );
                            proposals[proposals.length - 1].qtyOfTokensAcq.push(
                                ERC20(tokenAddresses[_assets[i]]).balanceOf(
                                    address(this)
                                )
                            );
                        }
                        if (shortTokenAddresses[_assets[i]] != address(0)) {
                            uint allocation = (lastSnapshotEth *
                                _percentage[i]) / 100;
                            ERC20short(shortTokenAddresses[_assets[i]]).mint(
                                address(this),
                                allocation
                            );
                            proposals[proposals.length - 1].qtyOfTokensAcq.push(
                                ERC20short(shortTokenAddresses[_assets[i]])
                                    .balanceOf(address(this))
                            );
                        }
                    }
                }
            }

            Proposal storage newProposal = proposals.push();
        }
        if (proposals.length == 1) {
            // 1. If first Proposal, convert all Eth to WETH

            WETH9(WETH).deposit{value: address(this).balance}();

            lastSnapshotEth = WETH9(WETH).balanceOf(address(this));

            // Snapshot captured of WETH at beggining of proposal w/ timestamp
            proposals[proposals.length - 1].startTime = block.timestamp;
            proposals[proposals.length - 1].startEth = lastSnapshotEth;

            /**
      The original, unsuccessful approach

      (bool success, ) = WETH9(WETH).call{value: address(this).balance}(abi.encodeWithSignature("deposit()"));
      require(success, "The transaction failed");
      console.log("this is an error");
      (bool go, bytes memory output) = tokenAddresses["WETH"].call(abi.encodeWithSignature("balanceOf(address)", address(this)));
      require(go);
      console.log(go);
      uint balance = abi.decode(output, (uint256));
      console.log("hello");
      console.log(balance);
      */
            // 2. Take asset weightings and purchase assets

            for (uint i = 0; i < _assets.length; i++) {
                assetWeightings[_assets[i]] = _percentage[i];
                proposals[proposals.length - 1].tokensWeightings.push(
                    _percentage[i]
                );
                proposals[proposals.length - 1].tokens.push(_assets[i]);

                if (_percentage[i] == 0) {
                    proposals[proposals.length - 1].qtyOfTokensAcq.push(0);
                } else {
                    if (tokenAddresses[_assets[i]] == WETH) {
                        uint allocation = (lastSnapshotEth * _percentage[i]) /
                            100;
                        proposals[proposals.length - 1].qtyOfTokensAcq.push(
                            allocation
                        );
                    }
                    if (
                        _percentage[i] != 0 &&
                        (keccak256(abi.encodePacked(_assets[i])) !=
                            wethRepresentation)
                    ) {
                        if (tokenAddresses[_assets[i]] != address(0)) {
                            uint allocation = (lastSnapshotEth *
                                _percentage[i]) / 100;
                            _swap(
                                WETH,
                                tokenAddresses[_assets[i]],
                                allocation,
                                0,
                                address(this)
                            );
                            proposals[proposals.length - 1].qtyOfTokensAcq.push(
                                ERC20(tokenAddresses[_assets[i]]).balanceOf(
                                    address(this)
                                )
                            );
                        } else if (
                            shortTokenAddresses[_assets[i]] != address(0)
                        ) {
                            uint allocation = (lastSnapshotEth *
                                _percentage[i]) / 100;
                            ERC20short(shortTokenAddresses[_assets[i]]).mint(
                                address(this),
                                allocation
                            );
                            proposals[proposals.length - 1].qtyOfTokensAcq.push(
                                ERC20short(shortTokenAddresses[_assets[i]])
                                    .balanceOf(address(this))
                            );
                        }
                    }
                }
            }
            Proposal storage newProposal = proposals.push();
        }
    }

    //token in = the token address you want to trade out of
    //token out = the token address you want as the output of this trade
    //amount in = the amount of tokens you are sending in
    //amount out Min = the minimum amount of tokens you want out of the trade
    //to = the address you want the tokens to be sent to

    function _swap(
        address _tokenIn,
        address _tokenOut,
        uint256 _amountIn,
        uint256 _amountOutMin,
        address _to
    ) public {
        // Calling ERC20 approve you allow the uniswap contract to spend the tokens in this contract
        ERC20(_tokenIn).approve(UNISWAP_V2_ROUTER, _amountIn);

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

        IUniswapV2Router(UNISWAP_V2_ROUTER).swapExactTokensForTokens(
            _amountIn,
            _amountOutMin,
            path,
            _to,
            block.timestamp
        );
    }

    function getHoldingsDataOfProposal(
        uint _index
    )
        public
        view
        returns (
            string[10] memory,
            uint[10] memory,
            uint[10] memory,
            uint[2] memory
        )
    {
        // Get Holdings data allows the input of an index of a proposal to get all the data from the Struct
        string[10] memory _tokens = [
            "WETH",
            "BAT",
            "WBTC",
            "UNI",
            "USDT",
            "sWETH",
            "sBAT",
            "sWBTC",
            "sUNI",
            "sUSDT"
        ];
        uint[10] memory actualHoldings;
        uint[10] memory fundAssetWeightings;
        uint[2] memory startingAndEndingTimes;

        startingAndEndingTimes[0] = proposals[_index].startTime;
        startingAndEndingTimes[1] = proposals[_index].endTime;

        fundAssetWeightings[0] = proposals[_index].tokensWeightings[0];
        actualHoldings[0] = proposals[_index].qtyOfTokensAcq[0];

        console.log(proposals[_index].tokens.length);

        for (uint i = 1; i < _tokens.length; i++) {
            fundAssetWeightings[i] = proposals[_index].tokensWeightings[i];
            if (
                tokenAddresses[_tokens[i]] != address(0) &&
                tokenAddresses[_tokens[i]] != WETH &&
                proposals[_index].tokensWeightings[i] != 0
            ) {
                console.log(_tokens[i]);
                console.log(proposals[_index].qtyOfTokensAcq[i]);
                actualHoldings[i] = proposals[_index].qtyOfTokensAcq[i];
                console.log("long tokens");
            } else if (
                shortTokenAddresses[_tokens[i]] != address(0) &&
                proposals[_index].tokensWeightings[i] != 0
            ) {
                console.log(_tokens[i]);
                console.log(proposals[_index].qtyOfTokensAcq[i]);
                actualHoldings[i] = proposals[_index].qtyOfTokensAcq[i];
                console.log("short tokens");
            }
        }
        console.log(actualHoldings[0]);
        return (
            _tokens,
            actualHoldings,
            fundAssetWeightings,
            startingAndEndingTimes
        );
    }

    function lengthOfProposals() public view returns (uint256) {
        return proposals.length - 1;
    }

    modifier onlyMember() {
        require(
            balanceOf(msg.sender) > 0,
            "Must be a member to use DAO function"
        );
        _;
    }
}
