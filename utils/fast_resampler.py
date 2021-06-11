import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import time
import torchaudio
import torch
from scipy import special


class Resampler(torch.nn.Module):
    """
    Based on https://github.com/danpovey/filtering/blob/master/lilfilter/resampler.py
    """

    def __init__(self, input_sr, output_sr, dtype, num_zeros=64,
                 cutoff_ratio=0.95, filter='kaiser', beta=14.0):
        super().__init__()  # init the base class
        """
        Args:
          input_sr:  The input sampling rate, AS AN INTEGER..
              does not have to be the real sampling rate but should
              have the correct ratio with output_sr.
          output_sr:  The output sampling rate, AS AN INTEGER.
              It is the ratio with the input sampling rate that is
              important here.
          dtype:  The torch dtype to use for computations (would be preferrable to 
               set things up so passing the dtype isn't necessary)
          num_zeros: The number of zeros per side in the (sinc*hanning-window)
              filter function.  More is more accurate, but 64 is already
              quite a lot. The kernel size is 2*num_zeros + 1.
          cutoff_ratio: The filter rolloff point as a fraction of the
             Nyquist frequency.
          filter: one of ['kaiser', 'kaiser_best', 'kaiser_fast', 'hann']
          beta: parameter for 'kaiser' filter


        """
        assert isinstance(input_sr, int) and isinstance(output_sr, int)
        if input_sr == output_sr:
            self.resample_type = 'trivial'
            return

        def gcd(a, b):
            """ Return the greatest common divisor of a and b"""
            assert isinstance(a, int) and isinstance(b, int)
            if b == 0:
                return a
            else:
                return gcd(b, a % b)

        d = gcd(input_sr, output_sr)
        input_sr, output_sr = input_sr // d, output_sr // d

        assert dtype in [torch.float32, torch.float64]
        assert num_zeros > 3  # a reasonable bare minimum
        np_dtype = np.float32 if dtype == torch.float32 else np.float64

        assert filter in ['hann', 'kaiser', 'kaiser_best', 'kaiser_fast']

        if filter == 'kaiser_best':
            num_zeros = 64
            beta = 14.769656459379492
            cutoff_ratio = 0.9475937167399596
            filter = 'kaiser'
        elif filter == 'kaiser_fast':
            num_zeros = 16
            beta = 8.555504641634386
            cutoff_ratio = 0.85
            filter = 'kaiser'

        zeros_per_block = min(input_sr, output_sr) * cutoff_ratio

        blocks_per_side = int(np.ceil(num_zeros / zeros_per_block))

        kernel_width = 2 * blocks_per_side + 1

        # We want the weights as used by torch's conv1d code; format is
        #  (out_channels, in_channels, kernel_width)
        # https://pytorch.org/docs/stable/nn.functional.html
        weights = torch.tensor((output_sr, input_sr, kernel_width), dtype=dtype)

        window_radius_in_blocks = blocks_per_side

        times = (
                np.arange(output_sr, dtype=np_dtype).reshape((output_sr, 1, 1)) / output_sr -
                np.arange(input_sr, dtype=np_dtype).reshape((1, input_sr, 1)) / input_sr -
                (np.arange(kernel_width, dtype=np_dtype).reshape((1, 1, kernel_width)) - blocks_per_side))

        def hann_window(a):
            """
            hann_window returns the Hann window on [-1,1], which is zero
            if a < -1 or a > 1, and otherwise 0.5 + 0.5 cos(a*pi).
            This is applied elementwise to a, which should be a NumPy array.

            The heaviside function returns (a > 0 ? 1 : 0).
            """
            return np.heaviside(1 - np.abs(a), 0.0) * (0.5 + 0.5 * np.cos(a * np.pi))

        def kaiser_window(a, beta):
            w = special.i0(beta * np.sqrt(np.clip(1 - ((a - 0.0) / 1.0) ** 2.0, 0.0, 1.0))) / special.i0(beta)
            return np.heaviside(1 - np.abs(a), 0.0) * w

        if filter == 'hann':
            weights = (np.sinc(times * zeros_per_block)
                       * hann_window(times / window_radius_in_blocks)
                       * zeros_per_block / input_sr)
        else:
            weights = (np.sinc(times * zeros_per_block)
                       * kaiser_window(times / window_radius_in_blocks, beta)
                       * zeros_per_block / input_sr)

        self.input_sr = input_sr
        self.output_sr = output_sr

        assert weights.shape == (output_sr, input_sr, kernel_width)
        if output_sr == 1:
            self.resample_type = 'integer_downsample'
            self.padding = input_sr * blocks_per_side
            weights = torch.tensor(weights, dtype=dtype, requires_grad=False)
            self.weights = weights.transpose(1, 2).contiguous().view(1, 1, input_sr * kernel_width)

        elif input_sr == 1:
            self.resample_type = 'integer_upsample'
            self.padding = output_sr * blocks_per_side
            weights = torch.tensor(weights, dtype=dtype, requires_grad=False)
            self.weights = weights.flip(2).transpose(0, 2).contiguous().view(1, 1, output_sr * kernel_width)
        else:
            self.resample_type = 'general'
            self.reshaped = False
            self.padding = blocks_per_side
            self.weights = torch.tensor(weights, dtype=dtype, requires_grad=False)

        self.weights = torch.nn.Parameter(self.weights, requires_grad=False)

    @torch.no_grad()
    def forward(self, data):

        if self.resample_type == 'trivial':
            return data
        elif self.resample_type == 'integer_downsample':
            (minibatch_size, seq_len) = data.shape
            # will be shape (minibatch_size, in_channels, seq_len) with in_channels == 1
            data = data.unsqueeze(1)
            data = torch.nn.functional.conv1d(data,
                                              self.weights,
                                              stride=self.input_sr,
                                              padding=self.padding)
            # shape will be (minibatch_size, out_channels = 1, seq_len);
            # return as (minibatch_size, seq_len)
            return data.squeeze(1)

        elif self.resample_type == 'integer_upsample':
            data = data.unsqueeze(1)
            data = torch.nn.functional.conv_transpose1d(data,
                                                        self.weights,
                                                        stride=self.output_sr,
                                                        padding=self.padding)

            return data.squeeze(1)
        else:
            assert self.resample_type == 'general'
            (minibatch_size, seq_len) = data.shape
            num_blocks = seq_len // self.input_sr
            if num_blocks == 0:
                # TODO: pad with zeros.
                raise RuntimeError("Signal is too short to resample")
            # data = data[:, 0:(num_blocks*self.input_sr)]  # Truncate input
            data = data[:, 0:(num_blocks * self.input_sr)].view(minibatch_size, num_blocks, self.input_sr)

            # Torch's conv1d expects input data with shape (minibatch, in_channels, time_steps), so transpose
            data = data.transpose(1, 2)

            data = torch.nn.functional.conv1d(data, self.weights,
                                              padding=self.padding)

            assert data.shape == (minibatch_size, self.output_sr, num_blocks)
            return data.transpose(1, 2).contiguous().view(minibatch_size, num_blocks * self.output_sr)