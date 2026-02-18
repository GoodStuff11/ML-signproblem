
import ffsim
import inspect

# Check for linear operators
if hasattr(ffsim, "uccsd_restricted_linear_operator"):
    print("\n--- uccsd_restricted_linear_operator ---")
    try:
        print(inspect.getsource(ffsim.uccsd_restricted_linear_operator))
    except:
        print("Could not get source for uccsd_restricted_linear_operator")
else:
    print("uccsd_restricted_linear_operator not found")

