from train.data_provider import Data_Form

if __name__ == '__main__':
    # flnm = {key: [path, scale]}
    # where scale indicates: pixel * scale = meter
    flnms = {"test1": [".//data//test1.csv", 0.1],
             "test2": [".//data//test2.csv", 0.1]}
    Data_Form(flnms)

    f = open(".//data//index.bin", 'rb')
    line = f.readline()
    line = f.readline()
    line = f.readline().decode()
    line_list = line.split()[0].split(',')
    head, tail = int(line_list[1]), int(line_list[2])
    f_data = open(".//data//data.bin", 'rb')
    f_data.seek(head)
    info = f_data.read(tail-head)
    info = info.decode()
    print(info)
