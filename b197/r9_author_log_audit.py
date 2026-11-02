#!/usr/bin/env python3
"""B197 R9 author-reported Mamba-3 K26 logs and HLS report *independent parser*.
This script does NOT independently rerun Vivado, Vitis, physical board or HLS.
Fetch upstream SHA externally. Only small derived JSON/CSV are redistributed.
"""
from pathlib import Path
import csv,hashlib,json,re,statistics,math
ROOT=Path("upstream")
configs=["baseline","pragma","fixed_pwl","fixed_cordic","pragma_fixed_pwl","pragma_fixed_cordic"]
results=[]
samples={}
source={}
for name in configs:
  pf=ROOT/"important_results"/"fpga"/f"fpga_{name}.txt"
  ps=ROOT/"important_results"/"syn"/f"syn_{name}.rpt"
  ft=pf.read_text(errors="replace");st=ps.read_text(errors="replace")
  source[name]={"fpga_sha256":hashlib.sha256(pf.read_bytes()).hexdigest(),
                "syn_sha256":hashlib.sha256(ps.read_bytes()).hexdigest(),
                "fpga_github_path":str(pf.relative_to(ROOT)),
                "hls_report_github_path":str(ps.relative_to(ROOT))}
  match=re.search(r"kria fpga-only\s*:\s*(\d+)\s+calls;\s*([\d.]+)\s+msecs total time",ft)
  assert match,(name,"missing board call receipt")
  calls=int(match.group(1));total_ms=float(match.group(2))
  assert calls==10
  seqvals=[(int(a),float(b),float(c)) for a,b,c in re.findall(
    r"Seq\s+(\d+)\s+\|\s+fpga:\s*([\d.]+)\s+ms\s+\|\s+TPOT:\s*([\d.]+)\s+ms/step",ft)]
  assert len(seqvals)==10,(name,seqvals)
  assert [s[0] for s in seqvals]==list(range(10))
  assert abs(sum(v[1] for v in seqvals)-total_ms)<0.05,(name,total_ms,seqvals)
  for seq,ms,tpot in seqvals:
    assert abs(ms/64-tpot)<0.0002,(name,seq,ms,tpot)
  samples[name]=[r[1] for r in seqvals]
  ts=st.split("+ Latency:",1)[1]
  lat=re.search(r"\|\s*(\d+)\|\s*(\d+)\|\s*([\d.]+)\s+ms\|",ts)
  assert lat,(name,ts[:500])
  cycles=int(lat.group(1));lat_ms=float(lat.group(3));assert cycles>1_000_000
  utilpart=st.split("== Utilization Estimates",1)[1]
  match=re.search(r"\|\s*Total\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|",utilpart)
  assert match,(name,"missing HLS total resource")
  bram,dsp,ff,lut,uram=map(int,match.groups())
  clock=re.search(r"\|\s*ap_clk\s*\|\s*([\d.]+)\s*ns\|\s*([\d.]+)\s*ns\|",st)
  assert clock,name
  out_error=re.search(r"out\s+max\s*/\s*mean\s*/\s*rmse:\s*([\deE+.-]+)\s*/\s*([\deE+.-]+)\s*/\s*([\deE+.-]+)",ft)
  assert out_error,name
  res=dict(config=name,board_total_ms_10_invocations=total_ms,
           board_mean_ms_per_64_step_invocation=total_ms/10,
           board_derived_ms_per_step=total_ms/(10*64),
           board_invocation_min_ms=min(samples[name]),board_invocation_max_ms=max(samples[name]),
           board_invocation_stdev_ms=statistics.stdev(samples[name]),
           HLS_cycles_estimate=cycles,HLS_estimated_ms_per_64_step_invocation=lat_ms,
           HLS_estimated_ms_per_step=lat_ms/64,
           HLS_clock_target_ns=float(clock.group(1)),HLS_estimated_comb_ns=float(clock.group(2)),
           author_HLS_BRAM18=bram,author_HLS_DSP=dsp,author_HLS_FF=ff,
           author_HLS_LUT=lut,author_HLS_URAM=uram,
           author_reported_out_max_abs_error=float(out_error.group(1)),
           board_minus_HLS_estimate_ms=(total_ms/10)-lat_ms,
           all_10_board_tests_pass=("Kria Host PASS" in ft))
  assert res["all_10_board_tests_pass"]
  results.append(res)
# Independent arithmetic for 32-bit no-DMA bridge payload source contract
NHEADS=4;HDIM=64;STATE=32;MIMO=2;NSTEPS=64
state=NHEADS*HDIM*STATE
bc=MIMO*NHEADS*STATE
x=NHEADS*HDIM
a=NHEADS
proj=MIMO*NHEADS*HDIM
const=a+a+3*proj
perstep=3*bc+3*x+2*a
inputs=const+state+NSTEPS*perstep
outputs=NSTEPS*x+state
assert (state,bc,x,const,perstep,inputs,outputs)==(8192,256,256,1544,1544,108552,24576)
wire_bytes=(inputs+outputs)*4
rawboard=next(r for r in results if r["config"]=="baseline")
pragmaboard=next(r for r in results if r["config"]=="pragma")
paired=[a-b for a,b in zip(samples["baseline"],samples["pragma"])]
effect=statistics.mean(paired)
paired_sd=statistics.stdev(paired)
# Approximate 95% t interval for paired differences with 9 df.
paired_ci=[effect-2.262157*paired_sd/math.sqrt(10),effect+2.262157*paired_sd/math.sqrt(10)]
summary={
  "contract":"6 author-reported Kria board logs + 6 Vitis HLS synthesis report files (independent parser, NOT independent FPGA PPA)",
  "source_commit":"a380d1b8ccc503fe42459056d3627201d101927d",
  "original_license":"NO_EXPLICIT_LICENSE_FOUND; no redistributed source or bitstreams",
  "board_test_kind":"host write/read of synthetic pre-generated step tensors, not closed-loop LM decoding",
  "model_contract":{"heads":4,"head_dim":64,"d_state":32,"MIMO":2,"steps_per_call":64,
   "initial_state_float32":state,"fixed_input_float32":const,
   "step_input_float32":perstep,"streamed_input_float32":inputs,"streamed_output_float32":outputs,
   "host_logical_payload_bytes":wire_bytes},
  "fp32_pragmas_hls_estimated_latency_reduction_percent":100*(1-pragmaboard["HLS_estimated_ms_per_64_step_invocation"]/rawboard["HLS_estimated_ms_per_64_step_invocation"]),
  "fp32_pragmas_board_mean_latency_reduction_percent":100*(1-pragmaboard["board_mean_ms_per_64_step_invocation"]/rawboard["board_mean_ms_per_64_step_invocation"]),
  "paired_ms_baseline_minus_pragma_mean":effect,
  "paired_ms_baseline_minus_pragma_95t_CI_exploratory_not_randomized":paired_ci,
  "baseline_approx_effective_MB_per_s":wire_bytes/(rawboard["board_mean_ms_per_64_step_invocation"]*1000),
  "baseline_effective_us_per_32bit_word":rawboard["board_mean_ms_per_64_step_invocation"]*1000/(inputs+outputs),
  "report_only_subtraction_warning":"Board time minus HLS estimate is not measured interface time; includes compute variation, stalls, driver and scheduling.",
  "source_sha256":source,
  "rows":results,
  "no_independent_k26_implementation":True,
  "no_new_architecture":True
}
Path("B197_R9_logs").mkdir(exist_ok=True)
Path("B197_R9_logs/K26_report_audit.json").write_text(json.dumps(summary,indent=2))
with Path("B197_R9_logs/K26_6config_summary.csv").open("w",newline="") as f:
  wr=csv.DictWriter(f,fieldnames=results[0].keys());wr.writeheader();wr.writerows(results)
print("B197_R9_AUTHOR_LOG_6CONFIG_AUDIT_PASS",
      json.dumps({k:summary[k] for k in ["fp32_pragmas_hls_estimated_latency_reduction_percent",
                   "fp32_pragmas_board_mean_latency_reduction_percent",
                   "baseline_approx_effective_MB_per_s",
                   "baseline_effective_us_per_32bit_word"]}))
print("B197_R9_PAIRED_EFFECT",paired,paired_ci)
print("B197_R9_CONFIG_ROWS",json.dumps([{k:r[k] for k in ["config","board_derived_ms_per_step","HLS_estimated_ms_per_step","author_HLS_LUT","author_HLS_BRAM18","author_HLS_DSP"]} for r in results]))
