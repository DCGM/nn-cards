# author: Pavel Ševčík

from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--graph-build-config", help="Json graph build config.", required=True)
    parser.add_argument("--backbone-config", help="Json network config.", required=True)
    parser.add_argument("--head-config", help="Json head config.", required=True)
    parser.add_argument("--optimization-config", help="Json optimization config", required=True)

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()



if __name__ == "__main__":
    main()
