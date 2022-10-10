import random
from export_csv import CSVHandler


class Generator:
    def __init__(self, n_lines_range=(2, 10), n_pages=100000, page_resolution=(2000, 2000),
                 line_range_offset=(200, 30), next_line_offset=60):
        self.n_lines_range = n_lines_range
        self.n_pages = n_pages
        self.page_resolution = page_resolution
        self.line_range_offset = line_range_offset
        self.next_line_offset = next_line_offset

    def create_line(self, prev_line, first_line=True):
        if first_line:
            x_offset = random.randint(0, self.line_range_offset[0])
            y_offset = self.line_range_offset[1]
            upper_left_x = random.randint((int(self.page_resolution[0] / 2) - x_offset), int(self.page_resolution[0] / 2))  # |p1[x1,y1] -                 |
            upper_left_y = random.randint((self.page_resolution[1] - y_offset), self.page_resolution[1])                    # |            -               |
            lower_right_x = upper_left_x + x_offset                                                                         # |              -             |
            lower_right_y = upper_left_y - y_offset                                                                         # |                - p2[x2, y2]|
            line = [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
        else:
            line = [prev_line[0], prev_line[1] + self.next_line_offset, prev_line[2], prev_line[3] + self.next_line_offset]

        return line

    def create_page(self, page_idx, csv):
        lines = []
        lines_center = []
        first_line = True
        prev_line = []
        rng = random.randint(self.n_lines_range[0], self.n_lines_range[1])

        for line_idx in range(rng):
            line = self.create_line(prev_line, first_line)
            center_point = ((int((line[0] + line[2]) / 2)), int((line[1]+line[3]) / 2))
            lines_center.append(center_point)
            lines.append(line)
            first_line = False
            prev_line = line

        sorted_lines = sorted(lines, key=lambda x: -(x[1]))
        sorted_centers = sorted(lines_center, key=lambda x: -(x[1]))

        for idx, line in enumerate(sorted_lines):
            csv.add({
                'image_id': "page_" + str(page_idx),
                #'line_center': sorted_centers[idx],
                #'line_coords': line,
                'startX': line[0],
                'startY': line[1],
                'endX': line[2],
                'endY': line[3],
                'line_order': idx
            })

        return csv

    def create_artificial_dataset(self):
        csv = CSVHandler()
        for page_idx in range(self.n_pages):
            csv = self.create_page(page_idx, csv)
        csv.to_csv("./csv/reading_order_dataset.csv")


