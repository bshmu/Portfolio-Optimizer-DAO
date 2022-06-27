import React, { useState } from "react";
import type { NextPage } from "next";
import Head from "next/head";
import styles from "../styles/Home.module.css";
import { JoinDAOButton } from "../components/JoinButton";
import { Web3Button } from "../components/web3button";
import StaticTable from "../components/StaticTable";
import { useWeb3Context } from "../context";
import EditTable from "../components/EditTable";

const Home: NextPage = () => {
  const { address, lpBalance } = useWeb3Context();
  return (
    <div className={styles.container}>
      <Head>
        <title>Portfolio Optimizer DAO</title>
        <meta name="description" content="Portfolio Optimizer DAO" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="mt-4 flex justify-end">
        <Web3Button />
      </div>

      <div className="mt-4 flex justify-end font-bold">
       Balance of LP Tokens: {lpBalance} {" "} OPD
      </div>

      <main className={styles.main}>
        {lpBalance == "0" || lpBalance == "" || lpBalance == undefined ? (
          <h1 className={styles.title}>Welcome to Portfolio Optimizer DAO</h1>
        ) : (
          // <StaticTable />
          ""
        )}
       

        {/* add balance of lp tokens check */}
        {/* <hr className="border-black pb-24"/> */}
        <div className="mt-24 mb-24">{lpBalance == "0" || lpBalance == "" || lpBalance == undefined ? <JoinDAOButton /> :<div> <EditTable /></div>}</div>
        
      </main>

      <footer className={styles.footer}>
        <a href="/" target="_blank" rel="noopener noreferrer">
          Built by Portfolio Optimizer DAO
          <span className={styles.logo}></span>
        </a>
      </footer>
    </div>
  );
};

export default Home;
