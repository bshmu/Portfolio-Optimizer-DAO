
interface ButtonProps {
  buttonLabel: string
  onClickHandler: () => any
}


const Button = (buttonProps:ButtonProps) => {
  return (
    <button onClick={buttonProps.onClickHandler}
      className="rounded-xl bg-sky-500 w-48 h-12 hover:bg-sky-400 border-black border-2"
    >
      {buttonProps.buttonLabel}
    </button>
  );
};

export default Button;
