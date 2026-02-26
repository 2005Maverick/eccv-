"""Wrapper to run create_dataset_100k.py and capture output."""
import sys, os, traceback
sys.path.insert(0, ".")
os.chdir(r"C:\Users\Dell\Desktop\ECCV DATASET")

# Redirect stdout/stderr to file
out = open("scripts/run_100k_out.txt", "w", encoding="utf-8")
sys.stdout = out
sys.stderr = out

# Patch sys.argv
sys.argv = ["scripts/create_dataset_100k.py", 
            "--target", "100000", "--passes", "5",
            "--skip-counterfactuals", "--skip-translation"]

try:
    exec(open("scripts/create_dataset_100k.py", encoding="utf-8").read())
except SystemExit as e:
    out.write("\nSystemExit: %s\n" % e)
except Exception as e:
    out.write("\nFATAL ERROR: %s\n" % e)
    traceback.print_exc(file=out)

out.close()
