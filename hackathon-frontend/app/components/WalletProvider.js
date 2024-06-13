"use client";

import { useState, useEffect } from "react";
import WalletContext from "./walletContext";

function WalletProvider({ children }) {
  const [userAddress, setUserAddress] = useState(null);

  // On component mount, check if userAddress exists in localStorage
  useEffect(() => {
    const savedAddress = localStorage.getItem("userAddress");
    if (savedAddress) {
      setUserAddress(savedAddress);
    }
  }, []);

  return (
    <WalletContext.Provider value={{ userAddress, setUserAddress }}>
      {children}
    </WalletContext.Provider>
  );
}

export default WalletProvider;
