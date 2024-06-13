import { createContext } from "react";

const WalletContext = createContext({
  userAddress: null,
  setUserAddress: () => {},
});

export default WalletContext;
