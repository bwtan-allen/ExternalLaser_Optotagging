import unittest

import numpy as np

from plotting_funcs import (
    _unit_channel_ids,
    _waveform_channel_view,
    _waveform_color_limits,
)


class WaveformChannelViewTests(unittest.TestCase):
    def test_centers_large_probe_view_on_units_peak_channel(self):
        waveform = np.zeros((200, 384))
        waveform[100, 281] = -70
        channel_ids = np.array([f'CH{channel}' for channel in range(384)])

        waveform_view, displayed_ids = _waveform_channel_view(
            waveform, channel_ids
        )

        self.assertEqual((200, 150), waveform_view.shape)
        self.assertIn('CH281', displayed_ids)
        displayed_peak = np.flatnonzero(displayed_ids == 'CH281')[0]
        self.assertEqual(-70, waveform_view[100, displayed_peak])

    def test_keeps_all_channels_for_smaller_single_shank_probe(self):
        waveform = np.zeros((200, 64))
        channel_ids = np.array([f'electrode-{channel}' for channel in range(64)])

        waveform_view, displayed_ids = _waveform_channel_view(
            waveform, channel_ids
        )

        self.assertIs(waveform, waveform_view)
        np.testing.assert_array_equal(channel_ids, displayed_ids)

    def test_resolves_sparse_template_channel_ids(self):
        class Sparsity:
            unit_id_to_channel_indices = {7: np.array([1, 3])}

        class WaveformExtractor:
            channel_ids = np.array(['A0', 'A1', 'B0', 'B1'])
            sparsity = Sparsity()

        channel_ids = _unit_channel_ids(WaveformExtractor(), 7, 2)

        np.testing.assert_array_equal(np.array(['A1', 'B1']), channel_ids)

    def test_uses_neutral_limits_for_flat_waveform(self):
        self.assertEqual(
            (-0.1, 0.1),
            _waveform_color_limits(np.zeros((120, 32))),
        )

    def test_color_limits_straddle_zero_for_one_sided_waveforms(self):
        negative_limits = _waveform_color_limits(np.array([-20.0, -5.0]))
        positive_limits = _waveform_color_limits(np.array([5.0, 20.0]))

        self.assertLess(negative_limits[0], 0)
        self.assertGreater(negative_limits[1], 0)
        self.assertLess(positive_limits[0], 0)
        self.assertGreater(positive_limits[1], 0)


if __name__ == '__main__':
    unittest.main()