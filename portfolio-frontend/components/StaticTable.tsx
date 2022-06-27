const StaticTable = () => {
  // people apart of the dao see this table
  // people not apart of the dao see this table
  // pull data from the smart contract
  // call the length of the propsals
  // call get holdings data propsals
  // call the length - 0
  return (
    <div>
      <h1 className="text-black text-xl font-bold mb-12 text-center">
        Performance
      </h1>

      <table className="table-auto">
        <thead>
          <tr className="bg-gray-200">
            <th className="pr-32">Token</th>
            <th className="pr-8">Actual Quantity</th>
            <th className="pr-8">Model Weight</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>ETH</td>
            <td>10</td>
            <td>90%</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default StaticTable;
