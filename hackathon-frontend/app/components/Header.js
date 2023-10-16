"use client";

import ConnectWalletButton from "./ConnectWalletButton";
import { Abril_Fatface } from "@next/font/google";

const abrilFatface = Abril_Fatface({ subsets: ["latin"], weight: ["400"] });

function Header() {
  return (
    <div className="flex justify-between items-center p-5 shadow-md">
      <div className={abrilFatface.className}>
        <h1 className="text-3xl font-semibold text-gray-300">OptimizerDAO</h1>
      </div>
      <ConnectWalletButton />
    </div>
  );
}

export default Header;
