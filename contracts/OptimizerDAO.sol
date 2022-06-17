pragma solidity ^0.8.0;

import "hardhat/console.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract OptimizerDAO {
  // May be able to delete membersTokenCount as tally is taken care of in ERC contract
  mapping(address => uint) public membersTokenCount;
  address public LPTokenAddress;
  uint public tokenReserve;
  uint public treasuryEth;

  //
  mapping(string => uint) public assetWeightings;

  // Proposal struct of token, expected performance and confidence level.
  struct Proposal {
    uint date;
    // Maps Token (i.e 'btc') to array
    mapping(string => uint[]) tokenWeightings;
    // Maps Token string to array of total token amount
    mapping(string => uint[]) userWeightings;
    mapping(string => uint[]) confidenceLevel;
    mapping(string => uint) proposalFinalPerformance;
    mapping(string => uint) proposalFinalConfidence;
  }


  constructor(address _LPTokenAddress) {
    LPTokenAddress = _LPTokenAddress;
  }

  function joinDAO() public payable {
    // What is the minimum buy in for the DAO?
    require(msg.value > 1 ether);
    if (treasuryEth == 0) {

      // If there is nothing in the treasury, provide liquidity to treasury
      // LP tokens are initially provided on a 1:1 basis
      treasuryEth = msg.value;
      ERC20(LPTokenAddress).transferFrom(address(this), msg.sender, treasuryEth);
      tokenReserve = ERC20(LPTokenAddress).totalSupply() - ERC20(LPTokenAddress).balanceOf(address(this));

    } else {
      // DAO members token count is diluted as more members join / add Eth
      uint proportionOfTokens = (msg.value * tokenReserve) / treasuryEth;
      tokenReserve += proportionOfTokens;
      treasuryEth += msg.value;
      ERC20(LPTokenAddress).transferFrom(address(this), msg.sender, proportionOfTokens);
    }
  }

  function leaveDAO() public {
    uint tokenBalance = ERC20(LPTokenAddress).balanceOf(msg.sender);
    require(tokenBalance > 0);

    // User gets back the relative % of the
    uint ethToWithdraw = (tokenBalance / tokenReserve) * treasuryEth;
    ERC20(LPTokenAddress).transferFrom(msg.sender, address(this), tokenBalance);
    payable(msg.sender).transfer(ethToWithdraw);
    tokenReserve -= tokenBalance;
    treasuryEth -= ethToWithdraw;
  }

  function submbitVote(string[] memory _token, int[] memory _perfOfToken, uint[] memory _confidenceLevels) public onlyMember {
    // User inputs token they'd like to vote on, the expected performance of token over time period and their confidence level
    // Loop through each token in list and provide a +1 on list
    // If token is in proposal, include in Struct and output average for Performance & confidence levels
    require((_token.length == _perfOfToken.length) && (_perfOfToken.length == _confidenceLevels.length), "Arrays must be the same size");

    uint numberOfVoterTokens = ERC20(LPTokenAddress).balanceOf(msg.sender);
    for (uint i = 0; i < _token.length; i++) {
      // get each value out of array
      latestProposal.userWeightings[_token[i]].push(_perfOfToken[i]);
      latestProposal.tokenWeightings[_token[i]].push(numberOfVoterTokens[i]);
      latestProposal.confidenceLevel[_token[i]].push(_confidenceLevels[i]);

    }
  }

  function findTokenWeight(string memory _token) internal returns(uint) {
    uint sumOfLPForToken;

    uint numeratorToken;
    uint numeratorConfidence;

    for (uint i = 0; i < latestProposal.tokenWeightings[_token].length; i++) {
      sumOfLPForToken += latestProposal.tokenWeightings[_token][i];
      numeratorToken += numberOfVoterTokens * _perfOfToken;
      // numeratorConfidence += numberOfVoterTokens *
    }
    uint weightedAveragePerformance = numeratorToken / sumOfLPForToken;
    uint weightedAverageConfidence = numeratorConfidence / sumOfLPForToken;
    latestProposal.proposalFinalPerformance[_token] = weightedAveragePerformance;
    latestProposal.proposalFinalConfidence[_token] = weightedAverageConfidence;

  }

  function rebalance() public {
    // Function is hit by external source... Chainlink?
    // Asset weightings is adjusted based on model
  }

  function initiateTradesOnUniswap() public {

  }

  modifier onlyMember {
      require(ERC20(LPTokenAddress).balanceOf(msg.sender) > 0);
      _;
   }

}