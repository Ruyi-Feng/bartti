from train.data_provider import Data_Form

if __name__ == '__main__':
    flnms = {"test1": ".//data//test1.csv",
             "test2": ".//data//test2.csv"}
    # Data_Form(flnms)

    f = open(".//data//index.bin", 'r')
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line_list = line.split()[0].split(',')
    head, tail = int(line_list[1]), int(line_list[2])
    f_data = open(".//data//data.bin")
    f_data.seek(head)
    info = f_data.read(tail-head)
    print(info)
