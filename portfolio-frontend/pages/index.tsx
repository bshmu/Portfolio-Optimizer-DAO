import React, { useState } from "react";
import type { NextPage } from "next";
import Head from "next/head";
import styles from "../styles/Home.module.css";
import { JoinDAOButton } from "../components/JoinButton";
import { Web3Button } from "../components/web3button";
import Table from "../components/table";
import { useWeb3Context } from "../context";

const Home: NextPage = () => {
  const { address } = useWeb3Context();

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

      <main className={styles.main}>
        <h1 className={styles.title}>Welcome to Portfolio Optimizer DAO</h1>

        {/* add balance of lp tokens check */}
        <div className="mt-24">{!address ? <JoinDAOButton /> : ""}</div>
        <Table />
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
