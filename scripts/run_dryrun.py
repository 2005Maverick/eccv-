"""Wrapper to run dry_run.py and capture output to file."""
import sys, os
sys.path.insert(0, ".")
os.chdir(r"C:\Users\Dell\Desktop\ECCV DATASET")

# Redirect stdout/stderr to file
out = open("scripts/dryrun_result.txt", "w", encoding="utf-8")
sys.stdout = out
sys.stderr = out

try:
    exec(open("scripts/dry_run.py", encoding="utf-8").read())
except SystemExit:
    pass
except Exception as e:
    out.write("\nFATAL: %s\n" % e)
    import traceback
    traceback.print_exc(file=out)

out.close()
