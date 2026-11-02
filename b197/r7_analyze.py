#!/usr/bin/env python3
"""B197 R7 paired trace comparison. Input is native-only instrumented log."""
from pathlib import Path
import struct,math,re,json,csv
raw=Path("B197_logs/stage_bits.log")
lines=raw.read_text().splitlines()
records={}
for line in lines:
    if not (line.startswith("B197_FLOAT|") or line.startswith("B197_Q8|")):continue
    kind,side,phase,token,layer,encoded=line.split("|",5)
    k=(kind,phase,int(token),int(layer))
    assert side in ("C","K"),line[:80]
    x=records.setdefault(k,{})
    assert side not in x,k
    x[side]=encoded
phase_order=["embed","rms_att","q8_att","proj_q","proj_k","proj_v",
             "conv_q","conv_k","conv_v","norm_q","norm_k","recur_out",
             "att_residual","layer_x","last_layer","final_rms"]
rows=[]
for (kind,phase,token,layer),v in records.items():
    assert set(v)=={"K","C"},(kind,phase,token,layer,set(v))
    if kind=="B197_FLOAT":
        a=v["K"];b=v["C"]
        assert len(a)==len(b) and len(a)%8==0
        ns=len(a)//8
        u=[struct.unpack("<f",int(a[j:j+8],16).to_bytes(4,"little"))[0] for j in range(0,len(a),8)]
        c=[struct.unpack("<f",int(b[j:j+8],16).to_bytes(4,"little"))[0] for j in range(0,len(b),8)]
        different=[i for i in range(ns) if a[i*8:i*8+8]!=b[i*8:i*8+8]]
        diffs=[abs(x-y) for x,y in zip(u,c) if math.isfinite(x) and math.isfinite(y)]
        finites=[abs(x) for x in c if math.isfinite(x)]
        row=dict(kind="float",token=token,layer=layer,phase=phase,n=ns,bit_mismatch=len(different),
                 max_abs_diff=max(diffs) if diffs else None,
                 max_abs_cpu=max(finites) if finites else None,
                 first_bad_index=different[0] if different else None)
    else:
        a=v["K"];b=v["C"]
        assert len(a)==len(b) and len(a)%2==0
        different=[i for i in range(len(a)//2) if a[i*2:i*2+2]!=b[i*2:i*2+2]]
        row=dict(kind="int8",token=token,layer=layer,phase=phase,n=len(a)//2,bit_mismatch=len(different),
                 max_abs_diff=None,max_abs_cpu=None,first_bad_index=different[0] if different else None)
    rows.append(row)
rows.sort(key=lambda r:(r["token"],r["layer"],phase_order.index(r["phase"]) if r["phase"] in phase_order else 999))
Path("B197_logs/stage_comparison.json").write_text(json.dumps(rows,indent=2))
with Path("B197_logs/stage_comparison.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["kind","token","layer","phase","n","bit_mismatch","max_abs_diff","max_abs_cpu","first_bad_index"])
    w.writeheader();w.writerows(rows)
assert len(rows)>150,("too few stage pairs",len(rows))
for token in (1,101,207):
    subset=[r for r in rows if r["token"]==token]
    assert subset
    bad=[r for r in subset if r["bit_mismatch"]]
    print("B197_R7_TOKEN",token,"paired_stages",len(subset),"mismatched_stages",len(bad))
    for row in subset[:12]:
        print("B197_R7_EARLY",token,row["layer"],row["phase"],row["bit_mismatch"],row["max_abs_diff"])
    if bad:print("B197_R7_FIRST_BAD",bad[0])
print("B197_R7_STAGE_TRACE_AUDIT_PASS",len(rows))
