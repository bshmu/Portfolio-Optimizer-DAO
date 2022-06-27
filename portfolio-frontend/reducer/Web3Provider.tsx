import { ethers } from "ethers";

export type Web3ProviderState = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  provider: any;
  web3Provider: ethers.providers.Web3Provider | null | undefined;
  address: string | null | undefined;
  network: ethers.providers.Network | null | undefined;
  lpBalance: string | null | undefined;
  connect: (() => Promise<void>) | null;
  disconnect: (() => Promise<void>) | null;
  joinDAO: (() => Promise<void>) | null;
  submitProposal: (() => Promise<void>) | null;
};

export const web3InitialState: Web3ProviderState = {
  provider: null,
  web3Provider: null,
  address: null,
  network: null,
  connect: null,
  disconnect: null,
  joinDAO: null,
  submitProposal: null,
  lpBalance: null,
};

export type Web3Action =
  | {
      type: "SET_WEB3_PROVIDER";
      provider?: Web3ProviderState["provider"];
      web3Provider?: Web3ProviderState["web3Provider"];
      address?: Web3ProviderState["address"];
      network?: Web3ProviderState["network"];
    }
  | {
      type: "SET_ADDRESS";
      address?: Web3ProviderState["address"];
    }
  | {
      type: "SET_NETWORK";
      network?: Web3ProviderState["network"];
    }
  | {
      type: "RESET_WEB3_PROVIDER";
    }
  | {
      type: "SET_LP_BALANCE";
      lpBalance: Web3ProviderState["lpBalance"];
    };

export function web3Reducer(
  state: Web3ProviderState,
  action: Web3Action
): Web3ProviderState {
  switch (action.type) {
    case "SET_WEB3_PROVIDER":
      return {
        ...state,
        provider: action.provider,
        web3Provider: action.web3Provider,
        address: action.address,
        network: action.network,
      };
    case "SET_ADDRESS":
      return {
        ...state,
        address: action.address,
      };
    case "SET_NETWORK":
      return {
        ...state,
        network: action.network,
      };
    case "SET_LP_BALANCE":
      return {
        ...state,
        lpBalance: action.lpBalance,
      };
    case "RESET_WEB3_PROVIDER":
      return web3InitialState;
    default:
      throw new Error();
  }
}
