import sys
n = 0
for line in sys.stdin:
    if n > 5:
        break
    print(line, end="")