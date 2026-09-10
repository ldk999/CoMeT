# TAS-MoE Experiment Outputs

The `run_all.py` helper generates per-scenario CSV summaries inside this directory. Each CSV shares the schema required by the TAS-MoE paper (temperature, performance, bandwidth, refresh, and NoC metrics).

## Latest validation run

The following command was executed inside the repository root to regenerate traces and run all four scenarios:

```
python comet_tasmoe/scripts/run_all.py \
  --arch comet_tasmoe/configs/arch_8x8_32x32x8.yaml \
  --workload comet_tasmoe/configs/workload_mixtral_8x7b.yaml \
  --tas comet_tasmoe/configs/tas_params.yaml \
  --out comet_tasmoe/results
```

This produced CSV reports such as `baseline/workload_mixtral_8x7b.csv` with peak logic temperature of 72.5 °C, peak DRAM temperature of 69.93 °C, and an average throughput of 42.67 inferences per second.

For quick smoke validation we also re-ran the baseline scenario only via:

```
python comet_tasmoe/scripts/run_single.py \
  --arch comet_tasmoe/configs/arch_8x8_32x32x8.yaml \
  --workload comet_tasmoe/configs/workload_mixtral_8x7b.yaml \
  --tas comet_tasmoe/configs/tas_params.yaml \
  --scenario baseline \
  --out comet_tasmoe/results
```

Old results can be removed safely; rerunning either command above will regenerate deterministic outputs for the provided configuration bundle.
