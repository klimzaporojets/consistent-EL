import logging
import struct

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()

def _read_int16(file):
    return struct.unpack("H", file.read(2))[0]

def _read_int32(file):
    return struct.unpack("I", file.read(4))[0]

def _read_int64(file):
    return struct.unpack("Q", file.read(8))[0]

def _read_string(file):
    length = _read_int32(file)
    xxxx = file.read(length)
    return str(xxxx, 'utf-8')

def _write_int16(file, number):
    file.write(struct.pack("H", number))

def _write_int32(file, number):
    file.write(struct.pack("I", number))

def _write_int64(file, number):
    file.write(struct.pack("Q", number))

def _write_string(file, string):
    data = string.encode('utf-8')
    _write_int32(file, len(data))
    file.write(data)
    return 4 + len(data)

class LineFileWriter:

    def __init__(self, filename):
        self.file_dat = open(filename + '.dat', 'wb')
        self.file_idx = open(filename + '.idx', 'wb')
        self.offset = 4 + 8
        self.num = 0
        _write_int32(self.file_dat, 2020)           # magic number
        _write_int64(self.file_dat, self.num)       # number of items
        _write_int64(self.file_idx, self.offset)    # first entry

    def write(self, line):
        self.offset += _write_string(self.file_dat, line)
        _write_int64(self.file_idx, self.offset)
        self.num += 1
        return

    def close(self):
        self.file_dat.seek(4)
        _write_int64(self.file_dat, self.num)

        self.file_dat.close()
        self.file_dat = None
        self.file_idx.close()
        self.file_idx = None

    @staticmethod
    def convert(output_file, input_file):
        logger.info('Convert {} to {}'.format(input_file, output_file))
        writer = LineFileWriter(output_file)
        with open(input_file, 'r') as file:
            for line in file.readlines():
                writer.write(line)
        writer.close()

class LineFileReader:

    def __init__(self, filename):
        self.file_dat = open(filename + '.dat', 'rb')
        self.file_idx = open(filename + '.idx', 'rb')
        self.file_dat.seek(4)
        self.size = _read_int64(self.file_dat)

    def read(self, idx):
        self.file_idx.seek(idx * 8)
        offset = _read_int64(self.file_idx)
        self.file_dat.seek(offset)
        return _read_string(self.file_dat)

    def close(self):
        self.file_dat.close()
        self.file_dat = None
        self.file_idx.close()
        self.file_idx = None

