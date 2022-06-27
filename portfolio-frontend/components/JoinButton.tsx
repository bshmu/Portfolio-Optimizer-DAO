import React, { useState } from "react";
import { useWeb3Context } from "../context";

interface JoinProps {
  join: (() => Promise<void>) | null;
}

const JoinButton = ({ join }: JoinProps) => {

  return join ? (
    <button
      className="rounded-xl bg-sky-500 w-48 h-12 hover:bg-sky-400 border-black border-2"
      onClick={join}
    >
      Join DAO
    </button>
  ) : (
    <button>Loading...</button>
  );
};

export function JoinDAOButton() {
  const { joinDAO } = useWeb3Context();
  return <JoinButton join={joinDAO} />;
}
