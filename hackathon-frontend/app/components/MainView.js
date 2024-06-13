"use client";

import { useState } from "react";
import { Noto_Serif, Roboto } from "@next/font/google";

const notoSerif = Noto_Serif({ subsets: ["latin"] });
const roboto = Roboto({
  subsets: ["latin"],
  weight: ["400", "700"],
});

function MainView() {
  const [token, setToken] = useState("ETH"); // default to ETH
  const [percentage, setPercentage] = useState(null);

  return (
    <section className="mt-10 text-center text-gray-400">
      <div className={notoSerif.className}>
        <h2 className="text-5xl mb-20 font-bold text-gray-300">
          What is your expected return?
        </h2>
      </div>

      <div className={roboto.className}>
        <p className="text-4xl mb-6 text-gray-300">
          I expect
          <select
            value={token}
            onChange={(e) => setToken(e.target.value)}
            className="mx-4 border p-1 focus:ring-0 focus:border-transparent text-gray-900 outline-none rounded-xl"
          >
            <option value="ETH">ETH</option>
            <option value="UNI">UNI</option>
          </select>
          to go up by
          <input
            value={percentage}
            onChange={(e) => {
              if (e.target.value === "") {
                setPercentage(null);
              } else {
                setPercentage(
                  Math.min(Math.max(Number(e.target.value), 0), 100)
                );
              }
            }}
            type="number"
            min="0"
            max="100"
            className="w-24 mx-4 border rounded p-1 focus:ring-0 focus:border-transparent text-center text-gray-900 outline-none no-spinners"
          />
          %
        </p>
        <button className="mt-14 bg-blue-800 hover:bg-blue-900 text-gray-300 hover:text-gray-500 py-3 px-8 rounded-lg focus:ring-0 focus:border-transparent text-2xl">
          Submit View
        </button>
      </div>
    </section>
  );
}

export default MainView;
