"""Smoke tests for Phase 9 Part 10: ReceiverEffectsPlotter.

Covers: each of the 7 diagnostic plots is created as a non-empty PNG,
both individually and via plot_all(), including on empty input data.
"""

from __future__ import annotations

from iff_simulator.sensors.iff import IFFMode, MeasurementStatus, PD_MODEL_GAUSSIAN
from iff_simulator.sensors.iff.measurement import DecodedIFFMeasurement
from iff_simulator.sensors.iff.receiver_statistics import ReceiverStatisticsCollector
from iff_simulator.sensors.iff.track import IFFTrack, TrackStatus, FriendFoeStatus
from iff_simulator.visualization import ReceiverEffectsPlotter


def _measurement(seq=1, time=1.0, range_m=500.0):
    return DecodedIFFMeasurement(
        measurement_id=seq, time=time, target_id="T1", ownship_id="OWNSHIP", mode=IFFMode.MODE_S,
        range_m=range_m, azimuth_deg=10.0, elevation_deg=5.0, icao_address="A00001",
        authentication_result=False, identity="BLUE", mission=None,
        reply_status=MeasurementStatus.VALID, processing_delay=50.0, propagation_delay=2.0,
        arrival_time=time, sequence_number=seq, signal_strength=0.8,
    )


def _track(seq=1, time=1.0, status=TrackStatus.TENTATIVE):
    return IFFTrack(
        track_id=1, aircraft_id="T1", ownship_id="OWNSHIP", time=time, range_m=500.0,
        azimuth_deg=10.0, elevation_deg=5.0, relative_velocity=None, mode=IFFMode.MODE_S,
        reply_status=MeasurementStatus.VALID, mode_s_address="A00001", authentication_result=False,
        friend_foe_status=FriendFoeStatus.UNKNOWN, track_status=status, track_quality=0.5,
        last_update_time=time, sequence_number=seq, reply_type=None, confidence=0.5,
        signal_strength=0.8, propagation_delay=2.0,
    )


def _build_populated_plotter():
    measurements = [_measurement(seq=n, time=float(n), range_m=100.0 * n) for n in range(1, 6)]
    stats = ReceiverStatisticsCollector()
    stats.record_tick_load(1.0, 1)
    stats.record_tick_load(2.0, 3)
    stats.record_garbled(1.5)
    stats.record_false_reply(2.5)
    tracks = [
        _track(seq=1, time=1.0, status=TrackStatus.TENTATIVE),
        _track(seq=2, time=2.0, status=TrackStatus.CONFIRMED),
        _track(seq=3, time=3.0, status=TrackStatus.LOST),
    ]
    return ReceiverEffectsPlotter(measurements, stats, tracks)


def _assert_nonempty_png(path):
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.suffix == ".png"


def test_each_plot_produces_a_nonempty_png(tmp_path):
    plotter = _build_populated_plotter()
    paths = [
        plotter.plot_detection_probability_vs_range(tmp_path, PD_MODEL_GAUSSIAN, {"r_max": 1000.0}, 2000.0),
        plotter.plot_signal_strength_vs_range(tmp_path),
        plotter.plot_reply_delay_histogram(tmp_path),
        plotter.plot_receiver_load_vs_time(tmp_path),
        plotter.plot_track_status_timeline(tmp_path),
        plotter.plot_garbled_replies_timeline(tmp_path),
        plotter.plot_false_replies_timeline(tmp_path),
    ]
    assert len(paths) == 7
    for path in paths:
        _assert_nonempty_png(path)


def test_plot_all_produces_seven_files(tmp_path):
    plotter = _build_populated_plotter()
    paths = plotter.plot_all(tmp_path, PD_MODEL_GAUSSIAN, {"r_max": 1000.0}, 2000.0)
    assert len(paths) == 7
    for path in paths:
        _assert_nonempty_png(path)


def test_plots_survive_empty_input_data(tmp_path):
    plotter = ReceiverEffectsPlotter(measurements=[], statistics=ReceiverStatisticsCollector(), track_log=[])
    paths = plotter.plot_all(tmp_path, PD_MODEL_GAUSSIAN, {"r_max": 1000.0}, 2000.0)
    assert len(paths) == 7
    for path in paths:
        _assert_nonempty_png(path)
