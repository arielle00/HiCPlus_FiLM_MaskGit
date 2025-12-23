from straw import straw

valid = []

# Try numeric names 1–22
for i in range(1, 23):
    try:
        records = straw("NONE", "4DNFILS2HLXC.hic", str(i), str(i), "BP", 1000000)
        if records:
            valid.append(str(i))
    except:
        pass

# Try sex chromosomes
for name in ["X", "Y"]:
    try:
        records = straw("NONE", "4DNFILS2HLXC.hic", name, name, "BP", 1000000)
        if records:
            valid.append(name)
    except:
        pass

print("✅ Available chromosomes:", valid)
