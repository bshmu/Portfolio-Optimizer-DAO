const {
  time,
  loadFixture,
} = require("@nomicfoundation/hardhat-toolbox/network-helpers");
const { anyValue } = require("@nomicfoundation/hardhat-chai-matchers/withArgs");
const { expect } = require("chai");


describe("OptimizerDAO", function() {
  let OptimizerDAO;
  let optimizer;
  let owner;
  let addrs;

  beforeEach(async function() {
    [owner, ...addrs] = await ethers.getSigners();
    OptimizerDAO = await ethers.getContractFactory("OptimizerDAO");
    optimizer = await OptimizerDAO.deploy();
    await optimizer.deployed();
  });

  describe("joinDAO", function() {
    it("Should mint correct tokens on initial joining event", async function() {
      const initialDeposit = ethers.utils.parseEther("0.001");
      
      // Send ETH and join the DAO
      await optimizer.connect(owner).joinDAO({ value: initialDeposit });

      // Check the balance of the owner
      const balance = await optimizer.balanceOf(owner.address);
      expect(balance).to.equal(initialDeposit); // Assuming 1:1 minting ratio for the first deposit
      console.log("Balance of owner: ", ethers.utils.formatEther(balance));
    });

    // Add more tests for other scenarios...
  });
});