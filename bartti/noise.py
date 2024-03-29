import copy
import math
import numpy as np
import random
import torch
import typing

MaskScheme = typing.List[typing.Tuple[int, int]]


class Noise():
    def __init__(self, noise_rate: float = 0.3, del_rate: float = 0.1, msk_rate: float = 0.1, poisson_rate: int = 3, frame_interval: float = 0.03, max_span_len: int = 5, max_seq_len: int = 512) -> None:
        """
        noise_rate:
        ----------
        float = 0.5
        the input x is randomly select to add noise by noise_rate

        del_rate:
        --------
        float = 0.5
        when add noise by randomly delete vehicles of one frame,
        del_rate percent of vehicles will be deleted in one frame.

        msk_rate:
        --------
        float = 0.5
        when add noise by randomly mask vehicles of one frame,
        msk_rate percent of vehicles will be masked in one frame.
        """
        self.MAX_CAR_NUM = 80
        self.LONG_SCALE = 300
        self.LATI_SCALE = 100
        self.SIZE_SCALE = 20
        self.noise_rate = noise_rate
        self.del_rate = del_rate
        self.msk_rate = msk_rate
        self._poisson_rate = poisson_rate
        self._max_span_len = max_span_len
        self.max_seq_len = max_seq_len
        self._poisson_dist = self._build_poisson_dist()
        # self.head_mark = [15.0, 150.0, 150.0, 150.0, 15.0, 15.0]   # 此值存疑
        self._mask_token = [10.0, 120.0, 1.0, 1.0, 1.0, 1.0]
        self.replen_mark = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def _noise_one(self, x):
        """_noise_one 选del_rate的车按照泊松删掉
        帧间隔保留
        """
        seq_len = len(x)
        del_len = self._pocess_len(seq_len, self.del_rate)
        if del_len < self._poisson_rate:
            return x.copy()
        spans = self._gen_spans(del_len)
        n_spans = len(spans)
        n_possible_insert_poses = seq_len - sum(spans) - n_spans + 1
        abs_insert_poses = sorted(random.sample(
            range(n_possible_insert_poses), n_spans))
        del_scheme = self._distribute_insert_poses(abs_insert_poses, spans)
        del_scheme = self._random_add_one(del_scheme)
        return self._del(x, del_scheme)

    def _noise_two(self, x):
        """_noise_two 选msk_rate的车mask(替换掉)
        帧间隔保留
        """
        seq_len = len(x)
        del_len = self._pocess_len(seq_len, self.del_rate)
        if del_len < self._poisson_rate:
            return x.copy()
        spans = self._gen_spans(del_len)
        n_spans = len(spans)
        n_possible_insert_poses = seq_len - sum(spans) - n_spans + 1
        abs_insert_poses = sorted(random.sample(
            range(n_possible_insert_poses), n_spans))
        mask_scheme = self._distribute_insert_poses(abs_insert_poses, spans)
        mask_scheme = self._random_add_one(mask_scheme)
        return self._mask(x, mask_scheme)

    def _noise_three(self, x):
        """_noise_three直接删掉一整帧 (1/2 改ID)"""
        cur_frame = -1
        del_frame = random.randint(2, 5)
        change_ID = self._randoms_half()
        n_x = []
        ID_base = self.car_num + random.randint(0, 20)
        for line in x:
            cur_frame = line[0]
            if cur_frame != del_frame:
                n_x.append([line[0]] + [(line[1] + float(change_ID and (cur_frame > del_frame)) * ID_base)%self.MAX_CAR_NUM] + line[2:])
        return n_x

    def _noise_four(self, x):
        """_noise_four交换两帧位置"""
        cur_frame = -1
        ids = random.sample(range(2, 5), 2)
        min_id, max_id = sorted(ids)
        pre_frames = []
        inter_frames = []
        post_frames = []
        frame_1 = []
        frame_2 = []
        for line in x:
            cur_frame = line[0]
            if cur_frame == min_id:
                frame_1.append(line)
            elif cur_frame == max_id:
                frame_2.append(line)
            elif cur_frame < min_id:
                pre_frames.append(line)
            elif cur_frame > max_id:
                post_frames.append(line)
            else:
                inter_frames.append(line)
        return pre_frames + frame_2 + inter_frames + frame_1 + post_frames

    def _mask(self, tokens: typing.List[list], mask_scheme: MaskScheme) -> typing.List[list]:
        mask_scheme = dict(mask_scheme)
        masked_tokens = []
        current_span = 0
        for i, t in enumerate(tokens):
            if i in mask_scheme:
                masked_tokens.append(self._mask_token)
                current_span = mask_scheme[i]
            if current_span > 0:
                current_span -= 1
                continue
            masked_tokens.append(t)
        return masked_tokens

    def _del(self, tokens: typing.List[list], del_scheme: MaskScheme) -> typing.List[list]:
        del_scheme = dict(del_scheme)
        del_tokens = []
        current_span = 0
        for i, t in enumerate(tokens):
            if i in del_scheme:
                current_span = del_scheme[i]
            if current_span > 0:
                current_span -= 1
                continue
            del_tokens.append(t)
        # del_tokens = self._change_id(del_tokens)
        return del_tokens

    def _change_id(self, tokens: typing.List[list]) -> typing.List[list]:
        # 有问题，暂未启用
        change_ID = self._randoms_half()
        if change_ID:
            last_id_dict = {}  # {ori: change}
            curr_id_dict = {}
            last_frm = -1
            for i in range(len(tokens)):
                t = tokens[i]
                if t[0] != last_frm:
                    last_id_dict = curr_id_dict.copy()
                    curr_id_dict = {}
                    curr_id_dict.update({t[1]: t[1] + ID_base})
                    last_frm = t[0]
                if t[1] in last_id_dict:
                    curr_id_dict.update({t[0]: last_id_dict[t[0]]})
                    tokens[i][0] = last_id_dict[t[0]]
                else:
                    ID_base = self.car_num + random.randint(0, 20)
                    curr_id_dict.update({t[1]: t[1] + ID_base})
                    tokens[i][0] = t[0] + ID_base
        return tokens


    def _distribute_insert_poses(self, abs_insert_poses: typing.List[int], spans: typing.List[int]) -> MaskScheme:
        offset = 0
        mask_scheme = []
        for abs_insert_pos, span in zip(abs_insert_poses, spans):
            insert_pos = abs_insert_pos + offset
            mask_scheme.append((insert_pos, span))
            offset += span + 1
        return mask_scheme

    def _random_add_one(self, mask_scheme: MaskScheme) -> MaskScheme:
        should_add_one = random.random() < 0.5
        if should_add_one:
            mask_scheme = [(insert_pos + 1, span)
                           for insert_pos, span in mask_scheme]
        return mask_scheme

    def _gen_spans(self, should_mask_len: int) -> typing.List[int]:
        spans = self._poisson_dist.sample((should_mask_len,))
        spans_cum = torch.cumsum(spans, dim=0)
        idx = torch.searchsorted(spans_cum, should_mask_len).item()
        spans = spans[:idx].tolist()
        if idx > spans_cum.size(0) - 1:
            return spans
        if idx - 1 < 0:
            return [self._poisson_rate]
        last = should_mask_len - spans_cum[idx - 1].item()
        if last > 0:
            spans.append(last)
        return spans

    def _pocess_len(self, seq_len: int, pocess_rate: float) -> int:
        x = seq_len * pocess_rate
        integer_part = int(x)
        fractional_part = x - float(integer_part)
        should_add = random.random() < fractional_part
        should_mask_len = integer_part + should_add
        return should_mask_len

    def _build_poisson_dist(self) -> torch.distributions.Categorical:
        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-self._poisson_rate)
        k_factorial = 1
        ps = []
        for k in range(0, self._max_span_len + 1):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= self._poisson_rate
            k_factorial *= k + 1
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        return torch.distributions.Categorical(ps)

    def _add_noise(self, x) -> list:
        """
        x: frame, id, xywh
        - _noise_one 选del_rate的车挖掉一帧 (1/2 改ID)
        - _noise_two 选msk_rate的车mask(替换掉)
        - _noise_three直接删掉一整帧 (1/2 改ID)
        - _noise_four交换两帧位置
        """
        self.car_num = int(len(x) / 5 - 1)
        self.trans_type = self._trans_type()
        # print("trans_type", trans_type)
        if self.trans_type == 0:
            return self._noise_one(x)
        elif self.trans_type == 1:
            return self._noise_two(x)
        elif self.trans_type == 2:
            return self._noise_three(x)
        elif self.trans_type == 3:
            return self._noise_four(x)

    def _trans_type(self) -> int:
        return random.randint(0, 3)

    def _randoms_half(self) -> bool:
        return random.random() < 0.5

    def _add_head_mark(self, x: list) -> list:
        # 暂未启用 headmark
        x_c = copy.deepcopy(x)
        if len(x_c) > 0:
            x_c.pop()
        return [self.head_mark] + x_c

    def _if_noise(self) -> bool:
        return random.random() < self.noise_rate

    def _stand(self, sec):
        if len(sec) < 1:
            return sec
        sec = np.array(sec)
        sec[:, 0] = sec[:, 0] - min(sec[:, 0]) + 1
        # print("sec", sec)
        # print("min sec", min(sec[:, 0]))
        sec[:, 1] = np.mod(sec[:, 1], self.MAX_CAR_NUM)+ 1
        sec[:, 2] = sec[:, 2] / self.LONG_SCALE
        sec[:, 3] = sec[:, 3] / self.LATI_SCALE
        sec[:, 4] = sec[:, 4] / self.SIZE_SCALE
        sec[:, 5] = sec[:, 5] / self.SIZE_SCALE
        return sec.tolist()

    def _comp(self, sec):
        sec = np.array(sec)
        while len(sec) < self.max_seq_len:
            sec = np.row_stack((sec, self.replen_mark))
        return sec

    def _post_process(self, enc_x, dec_x, x) -> tuple:
        return self._comp(enc_x), self._comp(dec_x), self._comp(x)

    def _pre_process(self, sec):
        return self._stand(sec)

    def derve(self, x: typing.List[list]) -> typing.Tuple[list, list, list]:
        """
        x: frame, id, x, y, w, h

        return
        ------
        enc_x: np.array -> torch(size=[batch, seq_len, c_in])
        enc_mark: np.array -> torch(size=[batch, 5])

        enc_mark means how many lines in each frames in enc_x
        """
        x = self._pre_process(x)
        dec_x = copy.deepcopy(x)
        x_copy = copy.deepcopy(x)
        if self._if_noise():
            enc_x = self._add_noise(x_copy)
        else:
            enc_x = x_copy
        return self._post_process(enc_x, dec_x, x)
