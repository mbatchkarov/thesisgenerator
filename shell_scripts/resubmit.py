import iterpipes3 as iterpipes
import re

failed = [41,43,45,47,50,71,73,75,77,80,137,139,141,143,146,167,169,171,173,
          176,197,199,201,203,206,227,229,231,233,236,257,259,261,263,266]
print(failed)
pattern = 'qsub -N exp{0} go-classify-single.sh {0}'
print(len(failed))
for id in failed:
    print(int(id))
    c = iterpipes.cmd(pattern.format(int(id)))
    out = iterpipes.run(c)
    for line in out:
        print(line)