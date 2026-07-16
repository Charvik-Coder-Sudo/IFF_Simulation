"""Integration tests for Phase 10's AnalysisReportGenerator.

Drives the real pipeline end to end (World -> TargetSelector ->
InterrogationScheduler -> AirborneTransponder -> ReceiverEffectsPipeline
-> IFFTrackManager), exactly as `run_analysis.py` does, then verifies:
all 6 required CSVs and 9 required plots are produced and non-empty,
the engineering report is generated, same-seed determinism (byte-
identical CSV text across two independent runs), different-seed
divergence, and that Ground Truth is never mutated by any analysis call.
"""

from __future__ import annotations

import copy

from iff_simulator.domain import Aircraft, AircraftState, Scenario, Vector3
from iff_simulator.sensors.iff import (
    AirborneTransponder,
    IFFTrackManager,
    InterrogationScheduler,
    PD_MODEL_GAUSSIAN,
    ReceiverConfig,
    ReceiverEffectsPipeline,
    TargetSelector,
    TrackStatus,
)
from iff_simulator.simulation import SimulationClock, World
from iff_simulator.analysis import AnalysisReportGenerator, PipelineRunRecord

ZERO = Vector3(0.0, 0.0, 0.0)


def _build_scenario():
    aircraft = [
        Aircraft(aircraft_id="OWNSHIP"),
        Aircraft(
            aircraft_id="T1", identity="FRIEND", iff_capability="MODE_S_CAPABLE",
            mode_data={"enabled_modes": ["MODE_S"]},
        ),
    ]
    history = {
        "OWNSHIP": [AircraftState(time=float(t), position=ZERO, velocity=ZERO) for t in range(1, 40)],
        "T1": [
            AircraftState(time=float(t), position=Vector3(100 + t, 0, 0), velocity=Vector3(1, 0, 0))
            for t in range(1, 40)
        ],
    }
    return Scenario(aircraft, history)


def _run_pipeline(seed: int) -> PipelineRunRecord:
    scenario = _build_scenario()
    clock = SimulationClock(start_time=1.0, dt=1.0, end_time=35.0)
    world = World(
        scenario, clock, ownship_id="OWNSHIP", maximum_range=1000.0, beam_width=360.0,
        beam_height=180.0, interrogation_rate=1.0,
    )
    selector = TargetSelector(world)
    scheduler = InterrogationScheduler(world, selector)
    transponder = AirborneTransponder(scenario)
    config = ReceiverConfig(
        seed=seed, pd_model=PD_MODEL_GAUSSIAN, pd_params={"r_max": 300.0}, pfa=0.1, fruiting_rate=0.1,
        garble_window_s=0.00001,
    )
    pipeline = ReceiverEffectsPipeline(config=config, ownship_id="OWNSHIP", maximum_range_m=1000.0)
    track_manager = IFFTrackManager()

    interrogations, replies, tick_results = [], [], []

    def process(interrogation):
        if interrogation is None:
            return
        reply = transponder.receive(interrogation)
        ownship_position = world.ownship.position
        target_position = scenario.get_state(interrogation.target_id).position
        tick_result = pipeline.process_tick(
            interrogation, reply, ownship_position, target_position, world.current_time()
        )
        interrogations.append(interrogation)
        replies.append(reply)
        tick_results.append(tick_result)
        if tick_result.real_measurement is not None:
            track_manager.update(tick_result.real_measurement)
        for measurement in tick_result.false_alarm_measurements:
            track = track_manager.update(measurement)
            if track is not None and track.track_status == TrackStatus.LOST:
                pipeline.mark_phantom_resolved(measurement.target_id)

    process(scheduler.tick())
    while not clock.finished():
        world.step()
        process(scheduler.tick())

    return PipelineRunRecord(
        scenario=scenario,
        interrogations=interrogations,
        replies=replies,
        tick_results=tick_results,
        active_tracks=track_manager.get_active_tracks(),
        completed_track_summaries=track_manager.get_completed_track_summaries(),
        receiver_statistics=pipeline.statistics.snapshot(),
    )


REQUIRED_CSV_NAMES = {
    "performance_metrics", "confusion_matrix", "latency_statistics",
    "detection_statistics", "authentication_statistics", "track_statistics",
}


def test_write_csv_outputs_produces_all_six_non_empty_files(tmp_path):
    record = _run_pipeline(seed=1)
    generator = AnalysisReportGenerator(record)
    paths = generator.write_csv_outputs(tmp_path)
    assert set(paths.keys()) == REQUIRED_CSV_NAMES
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_write_plots_produces_all_nine_non_empty_pngs(tmp_path):
    record = _run_pipeline(seed=1)
    generator = AnalysisReportGenerator(record)
    paths = generator.write_plots(tmp_path, PD_MODEL_GAUSSIAN, {"r_max": 300.0}, 1000.0)
    assert len(paths) == 9
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_write_engineering_report_produces_markdown(tmp_path):
    record = _run_pipeline(seed=1)
    generator = AnalysisReportGenerator(record)
    path = generator.write_engineering_report(tmp_path)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Detection Probability" in text
    assert "ROC" in text


def test_same_seed_gives_byte_identical_csv_output(tmp_path):
    record_a = _run_pipeline(seed=99)
    record_b = _run_pipeline(seed=99)

    paths_a = AnalysisReportGenerator(record_a).write_csv_outputs(tmp_path / "a")
    paths_b = AnalysisReportGenerator(record_b).write_csv_outputs(tmp_path / "b")

    for name in REQUIRED_CSV_NAMES:
        assert paths_a[name].read_text(encoding="utf-8") == paths_b[name].read_text(encoding="utf-8")


def test_different_seed_gives_different_performance_metrics(tmp_path):
    record_a = _run_pipeline(seed=1)
    record_b = _run_pipeline(seed=2)

    metrics_a = AnalysisReportGenerator(record_a).compute_all()["performance_metrics"]
    metrics_b = AnalysisReportGenerator(record_b).compute_all()["performance_metrics"]

    assert metrics_a != metrics_b


def test_ground_truth_scenario_is_never_mutated_by_analysis():
    record = _run_pipeline(seed=1)
    before = {
        aircraft_id: copy.deepcopy(record.scenario.get_aircraft(aircraft_id))
        for aircraft_id in record.scenario.list_aircraft_ids()
    }

    generator = AnalysisReportGenerator(record)
    generator.compute_all()

    after = {
        aircraft_id: record.scenario.get_aircraft(aircraft_id)
        for aircraft_id in record.scenario.list_aircraft_ids()
    }
    assert before == after


def test_compute_all_returns_every_expected_key():
    record = _run_pipeline(seed=1)
    results = AnalysisReportGenerator(record).compute_all()
    assert set(results.keys()) == {
        "performance_metrics", "detection_statistics", "authentication_statistics",
        "track_statistics", "roc_curve", "identity_confusion_matrix",
        "authentication_confusion_matrix", "latency_breakdown",
    }
