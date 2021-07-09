def rstring(s):
    return s[::-1]
def rnum(i):
    return 1.0/float(i)

def rlist(l):
    l.reverse()
    return l

def rdict(d):
    e = {}
    for k in d.keys():
        e[d[k]] = k
    return e
