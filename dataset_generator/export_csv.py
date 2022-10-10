import csv


class CSVHandler(object):
    def __init__(self):
        self.headers = list()
        self.rows = list()

    def add(self, row):
        if hasattr(row, '_asdict'):
            value = row._asdict()
        elif hasattr(row, '__dict__'):
            value = row.__dict__
        elif isinstance(row, dict):
            value = row
        else:
            raise ValueError('Not supported row type: {}'.format(type(row)))

        for header in value.keys():
            if header not in self.headers:
                self.headers.append(header)

        self.rows.append(value)

    def to_csv(self, file_name):
        with open(file_name, 'a') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(self.headers)

            for row in self.rows:
                csv_writer.writerow([row.get(header, None) for header in self.headers])
