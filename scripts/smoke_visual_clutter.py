"""Week 1 smoke test: run visual-clutter (Rosenholtz Feature Congestion + Subband Entropy)
on the ~50 sample covers. Known risk: stale package (Aug 2023), hard-pinned old deps,
pyrtools C dependency. Goal is to find breakage now, not in Week 3.

Run: .venv/bin/python scripts/smoke_visual_clutter.py
"""
import glob
import time
import traceback
import warnings

from visual_clutter import Vlc

warnings.filterwarnings("ignore")

covers = sorted(glob.glob("data/sample_covers/*.jpg"))
print(f"{len(covers)} sample covers")

ok, fail = 0, 0
fc_vals, se_vals, times = [], [], []
first_err = None

for path in covers:
    t0 = time.time()
    try:
        clt = Vlc(path, numlevels=3, contrast_filt_sigma=1, color_pool_sigma=3,
                  output_dir="/tmp/vc_out", prefix="smoke")
        fc, _ = clt.getClutter_FC()          # Feature Congestion (scalar)
        se = clt.getClutter_SE()             # Subband Entropy (scalar)
        dt = time.time() - t0
        fc_vals.append(float(fc)); se_vals.append(float(se)); times.append(dt)
        ok += 1
        print(f"OK   {path.split('/')[-1]:24s} FC={float(fc):7.3f}  SE={float(se):6.3f}  {dt:5.2f}s")
    except Exception as e:
        fail += 1
        if first_err is None:
            first_err = traceback.format_exc()
        print(f"FAIL {path.split('/')[-1]:24s} {type(e).__name__}: {e}")

print()
print(f"success {ok}/{len(covers)}  fail {fail}")
if fc_vals:
    import statistics as st
    print(f"FC  min/median/max: {min(fc_vals):.3f} / {st.median(fc_vals):.3f} / {max(fc_vals):.3f}")
    print(f"SE  min/median/max: {min(se_vals):.3f} / {st.median(se_vals):.3f} / {max(se_vals):.3f}")
    print(f"per-image time median: {st.median(times):.2f}s  ->  ~52k covers = {st.median(times)*52478/3600:.1f} h single-thread")
if first_err:
    print("\n--- first traceback ---\n" + first_err)
