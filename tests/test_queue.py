"""Tests for InterrogationQueue and write_interrogations_csv."""

from __future__ import annotations

from pathlib import Path

import pytest

from iff_simulator.sensors.iff import (
    IFFMode,
    InterrogationMessage,
    InterrogationQueue,
    UplinkFormat,
    write_interrogations_csv,
)


def _message(sequence_number: int, target_id: str = "T1", time: float = 0.0) -> InterrogationMessage:
    return InterrogationMessage(
        time=time,
        sequence_number=sequence_number,
        ownship_id="OWNSHIP",
        target_id=target_id,
        mode=IFFMode.MODE_S,
        uplink_format=UplinkFormat.UF11,
        range_m=100.0,
        azimuth_deg=0.0,
        elevation_deg=0.0,
    )


def test_empty_queue_length_is_zero():
    queue = InterrogationQueue()
    assert len(queue) == 0
    assert queue.length() == 0


def test_empty_queue_peek_returns_none():
    queue = InterrogationQueue()
    assert queue.peek() is None


def test_empty_queue_dequeue_raises_index_error():
    queue = InterrogationQueue()
    with pytest.raises(IndexError):
        queue.dequeue()


def test_enqueue_increases_length():
    queue = InterrogationQueue()
    queue.enqueue(_message(1))
    assert len(queue) == 1
    queue.enqueue(_message(2))
    assert len(queue) == 2


def test_fifo_order_preserved():
    queue = InterrogationQueue()
    queue.enqueue(_message(1, target_id="A"))
    queue.enqueue(_message(2, target_id="B"))
    queue.enqueue(_message(3, target_id="C"))

    assert queue.dequeue().target_id == "A"
    assert queue.dequeue().target_id == "B"
    assert queue.dequeue().target_id == "C"


def test_peek_does_not_remove():
    queue = InterrogationQueue()
    queue.enqueue(_message(1, target_id="A"))
    assert queue.peek().target_id == "A"
    assert len(queue) == 1
    assert queue.peek().target_id == "A"


def test_dequeue_removes_and_returns_oldest():
    queue = InterrogationQueue()
    queue.enqueue(_message(1, target_id="A"))
    queue.enqueue(_message(2, target_id="B"))
    first = queue.dequeue()
    assert first.target_id == "A"
    assert len(queue) == 1


def test_clear_empties_queue():
    queue = InterrogationQueue()
    queue.enqueue(_message(1))
    queue.enqueue(_message(2))
    queue.clear()
    assert len(queue) == 0
    assert queue.peek() is None


def test_write_interrogations_csv(tmp_path: Path):
    messages = [_message(1, target_id="A", time=0.0), _message(2, target_id="B", time=1.0)]
    output_path = tmp_path / "interrogations.csv"

    result_path = write_interrogations_csv(messages, output_path)

    assert result_path == output_path
    content = output_path.read_text(encoding="utf-8").splitlines()
    # Phase 8.5 Part 1 appends Closing_Velocity/Relative_Velocity.
    assert content[0] == (
        "Time,Sequence,Ownship_ID,Target_ID,Mode,UF,Range,Azimuth,Elevation,"
        "Closing_Velocity,Relative_Velocity"
    )
    assert len(content) == 3  # header + 2 rows


def test_write_interrogations_csv_creates_parent_dirs(tmp_path: Path):
    output_path = tmp_path / "nested" / "dir" / "interrogations.csv"
    write_interrogations_csv([_message(1)], output_path)
    assert output_path.exists()


def test_write_interrogations_csv_empty_messages(tmp_path: Path):
    output_path = tmp_path / "interrogations.csv"
    write_interrogations_csv([], output_path)
    content = output_path.read_text(encoding="utf-8").splitlines()
    assert len(content) == 1  # header only
