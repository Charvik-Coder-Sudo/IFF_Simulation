"""Tests for Phase 9 Part 9: ReceiverStatistics / ReceiverStatisticsCollector.

Covers: counters, running averages, time-series history, snapshot
immutability, and CSV writer determinism (mirrors test_phase_8_5.py's
CSV determinism tests).
"""

from __future__ import annotations

import pytest

from iff_simulator.sensors.iff import ReceiverStatisticsCollector, write_receiver_statistics_csv


def test_fresh_collector_snapshot_is_all_zero():
    stats = ReceiverStatisticsCollector().snapshot()
    assert stats.replies_received == 0
    assert stats.replies_lost == 0
    assert stats.replies_garbled == 0
    assert stats.replies_fruited == 0
    assert stats.false_replies == 0
    assert stats.average_detection_probability == 0.0
    assert stats.average_signal_strength == 0.0
    assert stats.average_delay_us == 0.0
    assert stats.receiver_load == 0.0


def test_counters_increment_correctly():
    collector = ReceiverStatisticsCollector()
    collector.record_received()
    collector.record_received()
    collector.record_lost()
    collector.record_garbled(time=1.0)
    collector.record_fruited(time=2.0)
    collector.record_false_reply(time=3.0)

    stats = collector.snapshot()
    assert stats.replies_received == 2
    assert stats.replies_lost == 1
    assert stats.replies_garbled == 1
    assert stats.replies_fruited == 1
    assert stats.false_replies == 1


def test_average_detection_probability_is_running_mean():
    collector = ReceiverStatisticsCollector()
    collector.record_pd_roll(1.0)
    collector.record_pd_roll(0.5)
    collector.record_pd_roll(0.0)
    assert collector.snapshot().average_detection_probability == pytest.approx(0.5)


def test_average_signal_strength_and_delay_are_running_means():
    collector = ReceiverStatisticsCollector()
    collector.record_signal_strength(0.8)
    collector.record_signal_strength(0.4)
    collector.record_delay(100.0)
    collector.record_delay(50.0)
    stats = collector.snapshot()
    assert stats.average_signal_strength == pytest.approx(0.6)
    assert stats.average_delay_us == pytest.approx(75.0)


def test_receiver_load_is_average_replies_per_tick():
    collector = ReceiverStatisticsCollector()
    collector.record_tick_load(time=1.0, replies_processed=2)
    collector.record_tick_load(time=2.0, replies_processed=0)
    collector.record_tick_load(time=3.0, replies_processed=4)
    assert collector.snapshot().receiver_load == pytest.approx(2.0)


def test_history_time_series_recorded_in_order():
    collector = ReceiverStatisticsCollector()
    collector.record_garbled(time=1.0)
    collector.record_garbled(time=2.0)
    collector.record_false_reply(time=1.5)
    collector.record_fruited(time=3.0)
    collector.record_tick_load(time=1.0, replies_processed=1)

    assert collector.garbled_history == [1.0, 2.0]
    assert collector.false_reply_history == [1.5]
    assert collector.fruited_history == [3.0]
    assert collector.load_history == [(1.0, 1)]


def test_write_receiver_statistics_csv_deterministic(tmp_path):
    collector = ReceiverStatisticsCollector()
    collector.record_received()
    collector.record_lost()
    collector.record_pd_roll(0.9)
    stats = collector.snapshot()

    path_a = write_receiver_statistics_csv(stats, tmp_path / "a.csv")
    path_b = write_receiver_statistics_csv(stats, tmp_path / "b.csv")
    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")


def test_multiple_tick_loads_accumulate_history_in_order():
    collector = ReceiverStatisticsCollector()
    for t in range(5):
        collector.record_tick_load(time=float(t), replies_processed=t)
    assert collector.load_history == [(0.0, 0), (1.0, 1), (2.0, 2), (3.0, 3), (4.0, 4)]


def test_snapshot_is_a_frozen_dataclass_not_the_live_collector():
    collector = ReceiverStatisticsCollector()
    collector.record_received()
    snapshot_a = collector.snapshot()
    collector.record_received()
    snapshot_b = collector.snapshot()
    assert snapshot_a.replies_received == 1
    assert snapshot_b.replies_received == 2


def test_write_receiver_statistics_csv_has_expected_header(tmp_path):
    stats = ReceiverStatisticsCollector().snapshot()
    path = write_receiver_statistics_csv(stats, tmp_path / "stats.csv")
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "Replies_Received,Replies_Lost,Replies_Garbled,Replies_Fruited,"
        "False_Replies,Average_Detection_Probability,Average_Signal_Strength,"
        "Average_Delay_Us,Receiver_Load"
    )
