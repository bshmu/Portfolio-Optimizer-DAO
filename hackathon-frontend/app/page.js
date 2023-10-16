"use client";

import { useContext } from "react";
import WalletContext from "./components/walletContext";
import Header from "./components/Header";
import JoinDao from "./components/JoinDao";
import MainView from "./components/MainView";

// Mock function - to be replaced with real logic later
const userIsDaoMember = (userAddress) => {
  // do something to check if user is a member of the DAO
  // ...
  return false; // return true if a member, false otherwise
};

export default function Home() {
  const { userAddress } = useContext(WalletContext);
  const isConnected = Boolean(userAddress);
  const isDaoMember = userIsDaoMember(userAddress);

  return (
    <main className="bg-blue-950 min-h-screen">
      <div>
        <Header />
        {!isConnected && <JoinDao />}
        {isConnected && !isDaoMember && <JoinDao />}
        {isConnected && isDaoMember && <MainView />}
      </div>
    </main>
  );
}
