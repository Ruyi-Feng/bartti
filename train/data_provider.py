import pandas as pd
from collections import deque

class Cols:
    def __init__(self, frame="frame", car_id="car_id", left="left", top="top", right="right", bottom="bottom", width="width", height="height"):
        self.frame = frame
        self.car_id = car_id
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.width = width
        self.height = height

class Data_Form:
    """
    STRUCTURE OF FORMED DATA
    data
    ----
    frame_id, car_id, cx, cy, width, top
    Each line is a single vehicle position

    index
    -----
    mark_ID, head_byte, tail_byte
    Each line is a position mark of one training token
    """
    def __init__(self, flnms: dict):
        self.LEN = 5
        self.SEC_LEN = 100.0   # _<m>
        self.window = deque(maxlen=self.LEN)
        self.last_bytes = 0
        self.flnms = flnms
        for flnm in flnms:
            labels = flnms[flnm]["labels"]
            cols = Cols(**labels)
            self._run(flnm, cols)

    def _run(self, flnm: str, cols: object):
        data = pd.read_csv(self.flnms[flnm]["path"])
        if "right" in self.flnms[flnm]["labels"]:
            data["width"] = abs(data[cols.right] - data[cols.left])
            data["height"] = abs(data[cols.bottom] - data[cols.top])
        self.min_y = data[cols.top].min() - 10  # 用来把数据集的y规范到较小的范围
        f_data = open(".\data\data.bin", 'ab+')
        f_index = open(".\data\index.bin", 'ab+')
        self.scale = self.flnms[flnm]["scale"]
        self._init_window()
        data_s = self._split_data(data, cols)
        data_s = data_s.sort_values(by=["split", cols.frame, cols.car_id]).reset_index(drop=True)
        for sp, data in data_s.groupby(data_s["split"]):
            for ID, group in data.groupby(data[cols.frame]):
                byte_frame = 0
                for _, row in group.iterrows():
                    id, cx, cy, w, h = self._stand_boxes(row, cols)
                    row_s = "{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(ID, id, cx, cy, w, h)
                    row_s = row_s.encode()
                    write_len = f_data.write(row_s)
                    byte_frame += write_len
                notable, head, tail = self._update_window(byte_frame)
                if notable:
                    idx_s = "{:s},{:d},{:d}\n".format(self._get_only_id(flnm, ID, sp), head, tail).encode()
                    f_index.write(idx_s)

    def _split_data(self, data, cols):
        """
        把数据分成小片区的
        """
        data["split"] = data[cols.left] // (self.SEC_LEN / self.scale)
        return data

    def _stand_boxes(self, row, cols):
        id = row[cols.car_id]
        cx = row[cols.left] + 0.5 * row[cols.width]
        cy = row[cols.top] + 0.5 * row[cols.height] - self.min_y
        w = row[cols.width]
        h = row[cols.height]
        id = id % 1024
        cx, cy, w, h = cx * self.scale, cy * self.scale, w * self.scale, h * self.scale
        return id, cx, cy, w, h

    def _init_window(self):
        self.last_bytes += self._sum_window()
        self.window.clear()

    def _get_only_id(self, flnm, ID, sp):
        return flnm + str(ID) + str(sp)

    def _sum_window(self):
        added = 0
        for _ in range(len(self.window)):
            added += self.window[_]
        return added

    def _update_window(self, byte: int):
        notable, head, tail = False, None, None
        # 这里因为多个flnm中last_bytes是用了同一个向后叠加的，考虑到衔接处，需要在前面判断并pop
        if len(self.window) == self.LEN:
            self.last_bytes += self.window.pop()
        self.window.appendleft(byte)  # appendleft之前保证len(self.q)<=4
        if len(self.window) == self.LEN:
            head = self.last_bytes
            tail = head + self._sum_window()
            notable = True
        return notable, head, tail
