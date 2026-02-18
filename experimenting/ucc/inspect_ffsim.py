
import ffsim
import inspect

print("Attributes of ffsim:")
for name in dir(ffsim):
    if "ucc" in name.lower():
        print(name)

if hasattr(ffsim, "UCCSDOpRestricted"):
    print("\n--- UCCSDOpRestricted ---")
    print(inspect.getsource(ffsim.UCCSDOpRestricted))
else:
    print("\nUCCSDOpRestricted not found in ffsim.")

# Check for linear operators
if hasattr(ffsim, "uccsd_operator"):
    print("\n--- uccsd_operator ---")
    # inspect.getsource might not work if it's a compiled extension or something, usually it works for python
    try:
        print(inspect.getsource(ffsim.uccsd_operator))
    except:
        print("Could not get source for uccsd_operator")

