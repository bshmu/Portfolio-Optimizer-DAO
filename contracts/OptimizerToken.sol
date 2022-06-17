pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract OptimizerToken is ERC20 {
    uint constant _initial_supply = 1000000000 * (10**18);
    constructor(address _dao) ERC20("Optimizer DAO LP Token", "ODT") {
        _mint(_dao, _initial_supply);
    }
}
