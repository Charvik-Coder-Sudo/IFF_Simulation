# Phase 10 Analysis -- Run Engineering Report

Generated automatically from one `PipelineRunRecord` by
`AnalysisReportGenerator.write_engineering_report`. See
`iff_simulator/analysis/docs/` for the static architecture/metrics/
complexity documentation this report's numbers should be read alongside.

## Performance Metrics

| Metric | Value |
|---|---|
| Detection Probability | 0.9279 |
| False Alarm Rate | 0.0013 |
| Authentication Success Rate | 0.0000 |
| Reply Success Rate | 0.9279 |
| Decoder Success Rate | 0.9330 |
| Track Confirmation Rate | 0.0811 |
| Average Track Lifetime (s) | 150.5714 |
| Average Detection Range (m) | 143.40 |
| Maximum Detection Range (m) | 247.34 |
| Average Processing Delay (us) | 49.99 |
| Average Propagation Delay (us) | 0.47 |
| Average Receiver Delay (us) | 0.0000 |
| Average Track Update Delay (s) | 0.0000 |
| Average Total Delay (us) | 50.47 |
| Average Signal Strength | 0.9763 |
| Average SNR (dB, proxy) | 19.90 |

## ROC

- Area Under Curve: 0.9941
- Points swept: 50

## Confusion Matrices

- Identity accuracy: 1.0000
- Authentication accuracy: 0.0000
