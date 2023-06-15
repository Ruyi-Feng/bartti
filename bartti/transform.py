import math
import random
import torch
import typing

MaskScheme = typing.List[typing.Tuple[int, int]]


class Noise():
    def __init__(self, noise_rate: float = 0.5, del_rate: float = 0.5, msk_rate: float = 0.5, poisson_rate: int = 3):
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
        self.noise_rate = noise_rate
        self.del_rate = del_rate
        self.msk_rate = msk_rate
        self._poisson_rate = poisson_rate
        self._poisson_dist = self._build_poisson_dist()
        self.head_mark = [666.0, 666.0, 666.0, 666.0, 666.0]

    def _noise_one(self, x):
        """_noise_one 选del_rate的车挖掉一帧 (1/2 改ID)
        这里需要注意考虑别把帧间隔删掉了

        ############这个属实没看懂丁总写的啥#############
        其实我应该改成按照poisson_dict sample 的删掉
        """
        seq_len = len(x)
        del_len = self._pocess_len(seq_len, self.del_rate)
        if del_len < self._poisson_rate:
            return x.copy()
        spans = self._gen_spans(del_len)
        n_spans = len(spans)
        n_possible_insert_poses = seq_len - sum(spans) - n_spans + 1
        abs_insert_poses = sorted(self._rand.sample(
            range(n_possible_insert_poses), n_spans))
        mask_scheme = self._distribute_insert_poses(abs_insert_poses, spans)
        mask_scheme = self._random_add_one(mask_scheme)
        return self._mask(x, mask_scheme)

    def _noise_two(self, x):
        """_noise_two 选msk_rate的车mask(替换掉)
        mask功能应该一样可以抄吧
        注意帧间隔要改一下
        确认一下mask的序列的形式
        """
        seq_len = len(x)
        del_len = self._pocess_len(seq_len, self.del_rate)
        if del_len < self._poisson_rate:
            return x.copy()
        spans = self._gen_spans(del_len)
        n_spans = len(spans)
        n_possible_insert_poses = seq_len - sum(spans) - n_spans + 1
        abs_insert_poses = sorted(self._rand.sample(
            range(n_possible_insert_poses), n_spans))
        mask_scheme = self._distribute_insert_poses(abs_insert_poses, spans)
        mask_scheme = self._random_add_one(mask_scheme)
        return self._mask(x, mask_scheme)

    def _noise_three(self, x):
        """_noise_three直接删掉一整帧 (1/2 改ID)"""
        pass

    def _noise_four(self, x):
        """_noise_four交换两帧位置"""
        pass

    def _mask(self, tokens: typing.List[str], mask_scheme: MaskScheme) -> typing.List[str]:
        mask_scheme = dict(mask_scheme)
        masked_tokens = []
        current_span = 0
        for i, t in enumerate(tokens):
            if i in mask_scheme:
                masked_tokens.append(self._mask_token)
                current_span = mask_scheme[i] - 1
                continue
            if current_span > 0:
                current_span -= 1
                continue
            masked_tokens.append(t)
        return masked_tokens

    def _distribute_insert_poses(self, abs_insert_poses: typing.List[int], spans: typing.List[int]) -> MaskScheme:
        offset = 0
        mask_scheme = []
        for abs_insert_pos, span in zip(abs_insert_poses, spans):
            insert_pos = abs_insert_pos + offset
            mask_scheme.append((insert_pos, span))
            offset += span + 1
        return mask_scheme

    def _random_add_one(self, mask_scheme: MaskScheme) -> MaskScheme:
        should_add_one = self._rand.random() < 0.5
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
        - _noise_one 选del_rate的车挖掉一帧 (1/2 改ID)
        - _noise_two 选msk_rate的车mask(替换掉)
        - _noise_three直接删掉一整帧 (1/2 改ID)
        - _noise_four交换两帧位置
        """
        trans_type = self._trans_type()
        if trans_type == 0:
            return self._noise_one(x)
        elif trans_type == 1:
            return self._noise_two(x)
        elif trans_type == 2:
            return self._noise_three(x)
        elif trans_type == 3:
            return self._noise_four(x)

    def _trans_type(self) -> int:
        return random.randint(0, 3)

    def _add_head_mark(self, x: list) -> list:
        return self.head_mark + x.pop()

    def _if_noise(self) -> bool:
        return random.random() < self.noise_rate

    def derve(self, x) -> typing.Tuple[list, list, list]:
        x_copy = x.copy()
        dec_x = self._add_head_mark(x_copy)
        if self._if_noise():
            enc_x = self._add_noise(x_copy)
        else:
            enc_x = x.copy()
        return enc_x, dec_x, x
