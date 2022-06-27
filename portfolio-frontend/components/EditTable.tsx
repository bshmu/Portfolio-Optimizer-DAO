import React, { useState, useCallback, useEffect } from "react";
import { useWeb3Context } from "../context";
import abi from "../utils/abi.json";
import { ethers } from "ethers";

interface EditTableData {
  token: any;
  view: number;
  confidence: number;
  viewType: any;
  viewRelativeToken: any;
}

const EditTable = () => {
  const { web3Provider } = useWeb3Context();

  const [token, setToken] = useState<EditTableData | any>("");
  const [view, setView] = useState<EditTableData | any>();
  const [confidence, setConfidence] = useState<EditTableData | any>();
  const [viewType, setViewType] = useState<EditTableData | any>("");
  const [viewRelativeToken, setViewRelativeToken] = useState<EditTableData | any>("");

  const contractAddress = "0x9a0DcA515dB6d9A97804e8364F3eF9e5cA817E4c";

  // creating proposal with this and each will be sent as an array [] sent to the contract
  // submit vote
  // only show this table if apart of the dao
  // hide if not apart of the dao

  const submitProposal = useCallback(async () => {
    const signer = web3Provider?.getSigner();
    try {
      const writeContract = new ethers.Contract(contractAddress, abi, signer);
      const viewRelativeToken = viewType === "relative" ? token : ""
      const txn = await writeContract.submitVote(
        [token],
        [view],
        [confidence],
        [viewType],
        [viewRelativeToken]
      );
      console.log("viewRelativeToken", viewRelativeToken)
      const receipt = await txn.wait();
      console.log(receipt);
    } catch (err) {
      console.log(err);
    }
  }, []);

  useEffect(() => {
    setViewRelativeToken(viewType === "relative" ? token : "")
  },[viewRelativeToken])

  return (
    <div>
      <h1 className="text-black text-xl font-bold mb-12 text-center">
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
              <select onChange={(e) => setToken(e.target.value)}>
                <option value="WBTC">WBTC</option>
                <option value="WETH">WETH</option>
                <option value="UNI">UNI</option>
                <option value="BAT">BAT</option>
              </select>
            </td>
            <td>
              <input
                onChange={(e) => setView(e.target.value)}
                type="number"
                defaultValue={1}
                min={0}
                max={100}
                className="border-black border-2"
              />
            </td>
            <td className="flex justify-center">
              <select onChange={(e) => setConfidence(e.target.value)}>
                {/* sne as a number  */}
                <option value={0}>Not Confident</option>
                <option value={25}>Somewhat Confident</option>
                <option value={50}>Neutral</option>
                <option value={75}>Confident</option>
                <option value={100}>Fully Confident</option>
              </select>
            </td>
            <td>
              <select onChange={(e) => setViewType(e.target.value)}>
                <option value="absolute">Absolute</option>
                <option value="relative">Relative</option>
              </select>
            </td>
            <td>
              
                <input value={viewRelativeToken} disabled={true}/>
            </td>
          </tr>
        </tbody>
      </table>
      <div className="mt-12 flex justify-center">
        <button
          className="rounded-xl bg-sky-500 w-48 h-12 hover:bg-sky-400 border-black border-2"
          onClick={submitProposal}
        >
          Submit Proposal
        </button>
      </div>{" "}
    </div>
  );
};

export default EditTable;
