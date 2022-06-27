const EditTable = () => {
  return (
    <table className="table-auto">
      <thead>
        <tr className="bg-gray-200">
          <th className="pr-8">Token</th>
          <th className="pr-8">View</th>
          <th className="pl-8 pr-8">Confidence</th>
          <th>Weight</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ETH</td>
          <td>
            <input className="border-black border-2" />
          </td>
          <td className="flex justify-center">
            <select>
              <option value="10">10</option>
              <option value="30">30</option>
              <option value="90">90</option>
              <option value="100">100</option>
            </select>
          </td>
          <td>
            <input className="border-black border-2" />
          </td>
        </tr>
      </tbody>
    </table>
  );
};

export default EditTable;
