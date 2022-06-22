import React from 'react'
import { useWeb3Context } from '../context/'

interface ConnectProps {
  connect: (() => Promise<void>) | null
}
const ConnectButton = ({ connect }: ConnectProps) => {
  return connect ? (
    <button className="rounded-xl bg-sky-500 w-48 h-12 hover:bg-sky-400 border-black border-2" onClick={connect}>Connect Wallet</button>
  ) : (
    <button>Loading...</button>
  )
}

interface DisconnectProps {
  disconnect: (() => Promise<void>) | null
}

const DisconnectButton = ({ disconnect }: DisconnectProps) => {
  return disconnect ? (
    <button className="rounded-xl bg-sky-500 w-48 h-12 hover:bg-sky-400 border-black border-2" onClick={disconnect}>Disconnect Wallet</button>
  ) : (
    <button>Loading...</button>
  )
}

export function Web3Button() {
  const { web3Provider, connect, disconnect } = useWeb3Context()

  return web3Provider ? (
    <DisconnectButton disconnect={disconnect} />
  ) : (
    <ConnectButton connect={connect} />
  )
}