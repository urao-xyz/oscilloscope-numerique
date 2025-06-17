import numpy as np
import main


def test_apply_lowpass_filter_changes_signal():
    # generate a 50 Hz sine wave for 1 second
    t = np.linspace(0, 1, int(main.TAUX_ECHANTILLONNAGE), endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t)

    # set global signals for all channels
    main.signals = [signal.copy() for _ in range(main.NUM_CHANNELS)]

    # apply a low-pass filter with cutoff at 10 Hz
    main.apply_filter(10, filter_type="lowpass")

    for filtered in main.signals:
        assert len(filtered) == len(signal)
        # ensure the filtered signal differs from the original
        assert not np.allclose(filtered, signal)


def test_calculate_rms_known_signal():
    t = np.linspace(0, 1, int(main.TAUX_ECHANTILLONNAGE), endpoint=False)
    signal = np.sin(2 * np.pi * 5 * t)
    main.signals = [signal.copy() for _ in range(main.NUM_CHANNELS)]
    rms_values = main.calculate_rms()
    expected = np.sqrt(np.mean(signal ** 2))
    for rms in rms_values:
        assert np.isclose(rms, expected)
