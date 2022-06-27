import React, { useState, useCallback, useEffect } from "react";
import { useWeb3Context } from "../context";
import abi from "../utils/abi.json";
import { ethers } from "ethers";
import { toast } from "react-toastify";

interface EditTableData {
  token: any;
  view: number;
  confidence: number;
  viewType: any;
  viewRelativeToken: any;
}

const EditTable = () => {
  const { web3Provider } = useWeb3Context();

  const [token, setToken] = useState<EditTableData | any>();
  const [view, setView] = useState<EditTableData | any>();
  const [confidence, setConfidence] = useState<EditTableData | any>();
  const [viewType, setViewType] = useState<EditTableData | any>();
  const [viewRelativeToken, setViewRelativeToken] = useState<EditTableData | any>();
  const [loading, setLoading] = useState<any>(false);
  const contractAddress = "0x9a0DcA515dB6d9A97804e8364F3eF9e5cA817E4c";

  // creating proposal with this and each will be sent as an array [] sent to the contract
  // submit vote
  // only show this table if apart of the dao
  // hide if not apart of the dao

  const submitProposal = useCallback(async () => {
    const signer = web3Provider?.getSigner();
    try {
      const writeContract = new ethers.Contract(contractAddress, abi, signer);
      const viewRelativeToken = viewType === "relative" && viewType === "USDT" || viewType === "absolute"  ? "" : token;
      const txn = await writeContract.submitVote(
        [token],
        [view],
        [confidence],
        [viewType],
        [viewRelativeToken]
      );
      setViewRelativeToken(viewRelativeToken)
      setLoading(true);
      const receipt = await txn.wait();
      console.log(receipt);
      toast.success("Proposal submitted to the DAO");
      setLoading(false);
    } catch (err) {
      console.log(err);
    }
  }, []);

  useEffect(() => {
  }, [viewRelativeToken]);

  return (
    <div>
      <h1 className="text-black text-[100px] font-bold mb-12 text-center">
        Proposal
      </h1>
      <table className="table-auto">
        <thead>
          <tr className="bg-gray-200">
            {/* drop down of all tokens check chat  */}
            <th className="pr-8">Token</th>
            {/* // number between 0-100 should take a number  */}
            <th className="pr-8">View</th>
            <th className="pl-8 pr-8">Confidence</th>
            {/* Absolute or relative drop down all lower case string */}
            <th className=" pr-8">View Type</th>
            {/* this is a string if the view type is absolute thats a empty string if relative then string of the token if view type is relative make it a hardocded dropdown 
          exclude tether if relative  */}
            <th>View Relative Token</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <select onChange={(e) => setToken(e.currentTarget.value)}>
                <option value="WETH">WETH</option>
                <option value="BAT">BAT</option>
                <option value="WBTC">WBTC</option>
                <option value="UNI">UNI</option>
                <option value="USDT">USDT</option>
              </select>
            </td>
            <td>
              <input
                onChange={(e) => setView(e.currentTarget.value)}
                type="number"
                defaultValue={1}
                min={0}
                max={100}
                className="border-black border-2"
              />
            </td>
            <td className="flex justify-center">
              <select onChange={(e) => setConfidence(e.currentTarget.value)}>
                {/* sne as a number  */}
                <option value={0}>Not Confident</option>
                <option value={25}>Somewhat Confident</option>
                <option value={50}>Neutral</option>
                <option value={75}>Confident</option>
                <option value={100}>Fully Confident</option>
              </select>
            </td>
            <td>
              <select onChange={(e) => setViewType(e.currentTarget.value)}>
                <option value="absolute">Absolute</option>
                <option value="relative">Relative</option>
              </select>
            </td>
            <td>
              <input value={viewRelativeToken} disabled={true} />
            </td>
          </tr>
        </tbody>
      </table>
      <div className="mt-12 flex justify-center">
        <button
          className="rounded-xl bg-sky-500 w-48 h-12 hover:bg-sky-400 border-black border-2"
          onClick={submitProposal}
        >
          {loading ? (
            <div className="flex justify-center">
              <svg
                role="status"
                className="w-6 h-6 mr-2 text-gray-300 animate-spin fill-[#e1ecef] spin 3s linear infinite"
                viewBox="0 0 100 101"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
                  fill="currentColor"
                />
                <path
                  d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
                  fill="currentFill"
                />
              </svg>
            </div>
          ) : (
            "Submit Proposal"
          )}
        </button>
      </div>{" "}
    </div>
  );
};

export default EditTable;
