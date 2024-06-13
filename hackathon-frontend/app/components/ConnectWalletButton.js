"use client";

import { useState, useEffect, useContext } from "react";
import { ethers } from "ethers";
import WalletContext from "./walletContext";

function ConnectWalletButton() {
  const { userAddress, setUserAddress } = useContext(WalletContext);
  const [isHovered, setIsHovered] = useState(false);

  async function handleWalletClick() {
    // If already connected, disconnect on button click
    if (userAddress) {
      setUserAddress(null);
      window.ethereum.removeAllListeners(); // Optionally remove all listeners

      // Remove from localStorage upon disconnection
      localStorage.removeItem("userAddress");
      return;
    }

    let provider;
    let signer;

    if (window.ethereum == null) {
      alert("Please install a wallet such as MetaMask!");
      return;
    } else {
      provider = new ethers.BrowserProvider(window.ethereum);
      signer = await provider.getSigner();
    }

    try {
      const address = await signer.getAddress();
      setUserAddress(address);

      localStorage.setItem("userAddress", address);
    } catch (err) {
      console.error("Failed to get address from signer", err);
      alert(
        "Failed to connect wallet. Please make sure your wallet is unlocked."
      );
    }
  }

  useEffect(() => {
    if (window.ethereum) {
      window.ethereum.on("accountsChanged", (accounts) => {
        if (accounts.length === 0) {
          setUserAddress(null);

          // Remove from localStorage
          localStorage.removeItem("userAddress");
        } else {
          setUserAddress(accounts[0]);

          // Save to localStorage
          localStorage.setItem("userAddress", accounts[0]);
        }
      });
    }

    return () => {
      if (window.ethereum) {
        window.ethereum.removeAllListeners("accountsChanged");
      }
    };
  }, [setUserAddress]);

  return (
    <button
      onClick={handleWalletClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className="text-gray-400 border-gray-500 border-2 rounded-full px-6 py-2 hover:text-gray-600 focus:outline-none"
    >
      {userAddress
        ? isHovered
          ? "Disconnect"
          : userAddress.substring(0, 6) + "..." + userAddress.substring(38)
        : "Connect Wallet"}
    </button>
  );
}

export default ConnectWalletButton;
