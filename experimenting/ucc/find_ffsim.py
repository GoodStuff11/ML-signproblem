
import ffsim
import inspect
import os

print(f"ffsim file: {inspect.getfile(ffsim)}")
# Try to find uccsd_restricted_linear_operator in modules
for name, module in inspect.getmembers(ffsim):
    if inspect.ismodule(module):
        if hasattr(module, "uccsd_restricted_linear_operator"):
            print(f"Found in {name}")

# Also check if it is available in ffsim directly but I missed it
if "uccsd_restricted_linear_operator" in dir(ffsim):
    print("It is in ffsim dir")
else:
    print("Not in ffsim dir")

