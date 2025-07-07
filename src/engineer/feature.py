from argparse import ArgumentParser

# CLI Input Arguments
def cli_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Accepts command-line arguments required to process the data.")
    parser.add_argument("-r", "--raw", help="Raw dataset", required=True)
    parser.add_argument("-p", "--process", help="Raw dataset", required=True)
    
    args = parser.parse_args()
    
    return args

# Run data pipeline
def main() -> None:
    cli_parser = cli_arguments()
    print(cli_parser)


if __name__ == "__main__":
    main()