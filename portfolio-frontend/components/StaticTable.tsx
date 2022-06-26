const StaticTable = () => {
  return (
    <table className="table-auto">
      <thead>
        <tr className="bg-gray-200">
          <th className="pr-32">Token</th>
          <th className="pr-8">Designated Weight</th>
          <th className="pr-8">Actual Weight</th>
          <th>USD value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ETH</td>
          <td>10%</td>
          <td>90%</td>
          <td>$1000</td>
        </tr>
      </tbody>
    </table>
  );
};

export default StaticTable;
