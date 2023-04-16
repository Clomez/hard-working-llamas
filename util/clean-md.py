import re

filename = "../Huberman_index_01.md"

def clean():
    with open(filename, 'r') as f:
        regex = re.compile(r"(.)\1{9,}")
        raw_text = f.read()
        out_file = regex.sub(lambda m: m.group().replace('-',""), raw_text)

    try:
        f2 = open(f"huberman_clean.md", "w")

        wr = str(out_file) + "\n"
        f2.write(wr)
        f2.close()
    except:
        raise Exception("Cant write to file")
    
clean()