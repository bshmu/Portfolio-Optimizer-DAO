"use client";

import { useContext, useState } from "react";
import WalletContext from "./walletContext";
import ConnectWalletButton from "./ConnectWalletButton";
import { Noto_Serif, Roboto } from "@next/font/google";

const notoSerif = Noto_Serif({ subsets: ["latin"] });
const roboto = Roboto({
  subsets: ["latin"],
  weight: ["400", "700"],
});

function JoinDao({ onJoin }) {
  const { userAddress } = useContext(WalletContext);
  const [ethAmount, setEthAmount] = useState("");

  const handleJoin = () => {
    // Here, you can implement the logic for sending the ETH, then call the `onJoin` callback
    // onJoin();
  };

  const isConnected = Boolean(userAddress);

  return (
    <section className="mt-10 text-center text-gray-400">
      <div className={notoSerif.className}>
        <h2 className="text-6xl mb-20 font-bold text-gray-300">Join our DAO</h2>
      </div>
      <div className={roboto.className}>
        <p className="text-2xl mb-10 text-gray-300 w-1/2 mx-auto">
          To participate in our DAO, send an amount of ETH, it will be
          proportionally converted to DAO Governance tokens.
        </p>

        <div className="mb-10">
          <input
            value={ethAmount}
            onChange={(e) => {
              const value = parseFloat(e.target.value);
              if (value >= 0) {
                setEthAmount(e.target.value);
              }
            }}
            type="number"
            min="0.10"
            step="0.01"
            placeholder="min 0.10"
            className="w-40 mx-2 border rounded p-2 focus:ring-0 focus:border-transparent text-center text-gray-900 outline-none no-spinners "
            disabled={!isConnected}
          />
          <span className="text-2xl text-gray-300">ETH</span>
        </div>

        {isConnected ? (
          <button
            disabled={parseFloat(ethAmount) < 0.1}
            onClick={handleJoin}
            className="bg-blue-800 hover:bg-blue-900 text-gray-300 hover:text-gray-500 py-3 px-8 rounded-lg focus:ring-0 focus:border-transparent text-2xl "
          >
            Join
          </button>
        ) : (
          <ConnectWalletButton />
        )}
      </div>
    </section>
  );
}

export default JoinDao;
