from generator import Generator
from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--n-lines-range", nargs='+', type=int, help="Number of lines in given range, e.g. (2, 10)")
    parser.add_argument("--n-pages", type=int, help="Number of pages.")
    parser.add_argument("--page-resolution", nargs='+', type=int, help="Image page resolution.")
    parser.add_argument("--line-offset", nargs='+', type=int,
                    help="Offsets for x and y coordinates to specify line width and height, e.g (x_offset, y_offset).")
    parser.add_argument("--next-line-offset", help="Offset for y coordinate for the next line.", type=int)
    parser.add_argument("--save-path", help="Path to csv output.", type=str, default="./csv")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    g = Generator(tuple(args.n_lines_range), args.n_pages, tuple(args.page_resolution), tuple(args.line_offset),
                  args.next_line_offset, args.save_path)
    g.create_artificial_dataset()


if __name__ == "__main__":
    main()