import React, {useState} from "react";
import type { NextPage } from "next";
import Head from "next/head";
import styles from "../styles/Home.module.css";
import Button from "../components/button";


const Home: NextPage = () => {
  const joinDAO = () => {
    console.log("hiiii")
  }

  return (
    <div className={styles.container}>
      <Head>
        <title>Portfolio Optimizer DAO</title>
        <meta name="description" content="Portfolio Optimizer DAO" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>Welcome to Portfolio Optimizer DAO</h1>
        
        <div className="mt-48">
          <Button buttonLabel="JOIN" onClickHandler={joinDAO}/>
        </div>
        {/* <p className={styles.description}>
          Get started by editing{" "}
          <code className={styles.code}>pages/index.tsx</code>
        </p> */}

        {/* <div className={styles.grid}>
          <a href="https://nextjs.org/docs" className={styles.card}>
            <h2>Documentation &rarr;</h2>
            <p>Find in-depth information about Next.js features and API.</p>
          </a>

          <a href="https://nextjs.org/learn" className={styles.card}>
            <h2>Learn &rarr;</h2>
            <p>Learn about Next.js in an interactive course with quizzes!</p>
          </a>

          <a
            href="https://github.com/vercel/next.js/tree/canary/examples"
            className={styles.card}
          >
            <h2>Examples &rarr;</h2>
            <p>Discover and deploy boilerplate example Next.js projects.</p>
          </a>

          <a
            href="https://vercel.com/new?utm_source=create-next-app&utm_medium=default-template&utm_campaign=create-next-app"
            className={styles.card}
          >
            <h2>Deploy &rarr;</h2>
            <p>
              Instantly deploy your Next.js site to a public URL with Vercel.
            </p>
          </a>
        </div> */}
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
