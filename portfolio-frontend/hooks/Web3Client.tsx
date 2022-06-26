import { useEffect, useReducer, useCallback } from "react";
import { ethers } from "ethers";
import Web3Modal from "web3modal";
import WalletConnectProvider from "@walletconnect/web3-provider";
import abi from "../utils/abi.json";

import {
  Web3ProviderState,
  Web3Action,
  web3InitialState,
  web3Reducer,
} from "../reducer";

import { toast } from "react-toastify";

const providerOptions = {
  walletconnect: {
    package: WalletConnectProvider, // required
    options: {
      infuraId: "2dea9cadae1f45f7930e57184ada09e6",
    },
  },
};

let web3Modal: Web3Modal | null;
if (typeof window !== "undefined") {
  web3Modal = new Web3Modal({
    network: "mainnet", // optional
    cacheProvider: true,
    providerOptions, // required
  });
}

export const useWeb3 = () => {
  const [state, dispatch] = useReducer(web3Reducer, web3InitialState);
  const { provider, web3Provider, address, network, lpBalance } = state;

  const contractAddress = "0xEA4Ac9058B8C3f615768Fb5E8EBeacc780e6b6a5";
  const RPC_URL =
    "https://rinkeby.infura.io/v3/2dea9cadae1f45f7930e57184ada09e6";

  const connect = useCallback(async () => {
    if (web3Modal) {
      try {
        const provider = await web3Modal.connect();
        const web3Provider = new ethers.providers.Web3Provider(provider);
        const signer = web3Provider.getSigner();
        const address = await signer.getAddress();
        const network = await web3Provider.getNetwork();
        toast.success("Connected to Web3");

        dispatch({
          type: "SET_WEB3_PROVIDER",
          provider,
          web3Provider,
          address,
          network,
        } as Web3Action);
        tokenBalance(address);
      } catch (e) {
        console.log("connect error", e);
      }
    } else {
      console.error("No Web3Modal");
    }
  }, []);

  const disconnect = useCallback(async () => {
    if (web3Modal) {
      web3Modal.clearCachedProvider();
      if (provider?.disconnect && typeof provider.disconnect === "function") {
        await provider.disconnect();
      }
      toast.error("Disconnected from Web3");
      dispatch({
        type: "RESET_WEB3_PROVIDER",
      } as Web3Action);
    } else {
      console.error("No Web3Modal");
    }
  }, [provider]);

  const joinDAO = useCallback(async () => {
    try {
      const signer = web3Provider?.getSigner();
      const writeContract = new ethers.Contract(contractAddress, abi, signer);
      const txn = await writeContract.joinDAO({
        value: ethers.utils.parseEther("0.1")._hex,
      });

      const receipt = await txn.wait();
      console.log(receipt);
      toast.success("Welcome to the DAO");
    } catch (e) {
      console.log("Error joining DAO", e);
      console.error("Unable to join DAO");
    }
  }, []);

  const tokenBalance = useCallback(async (address: string) => {
    try {
      const signer = state.web3Provider?.getSigner();
      const readContract = new ethers.Contract(
        contractAddress,
        abi,
        signer
      );
      const balanceOf = await readContract.balanceOf(address) / 10 ** 18;

      dispatch({
        type: "SET_LP_BALANCE",
        lpBalance: balanceOf.toString(),
      } as Web3Action);

    } catch (err) {
      console.log(err);
    }
  }, []);

  useEffect(() => {}, [lpBalance]);

  // Auto connect to the cached provider
  useEffect(() => {
    if (web3Modal && web3Modal.cachedProvider) {
      connect();
    }
  }, [connect]);

  useEffect(() => {
    if (provider?.on) {
      const handleAccountsChanged = (accounts: string[]) => {
        toast.info("Changed Web3 Account");
        dispatch({
          type: "SET_ADDRESS",
          address: accounts[0],
        } as Web3Action);
      };

      const handleChainChanged = (_hexChainId: string) => {
        if (typeof window !== "undefined") {
          console.log("switched to chain...", _hexChainId);
          toast.info("Web3 Network Changed");
          window.location.reload();
        } else {
          console.log("window is undefined");
        }
      };

      const handleDisconnect = (error: { code: number; message: string }) => {
        // eslint-disable-next-line no-console
        console.log("disconnect", error);
        disconnect();
      };

      provider.on("accountsChanged", handleAccountsChanged);
      provider.on("chainChanged", handleChainChanged);
      provider.on("disconnect", handleDisconnect);

      // Subscription Cleanup
      return () => {
        if (provider.removeListener) {
          provider.removeListener("accountsChanged", handleAccountsChanged);
          provider.removeListener("chainChanged", handleChainChanged);
          provider.removeListener("disconnect", handleDisconnect);
        }
      };
    }
  }, [provider, disconnect]);

  return {
    provider,
    web3Provider,
    address,
    network,
    lpBalance,
    connect,
    disconnect,
    joinDAO,
  } as Web3ProviderState;
};
