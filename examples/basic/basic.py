from cuTensor import cuTensor as T

t = T([4, 5, 2])
t.fill()
t.info()

t2= T([2, 5, 2])
t2.fill(2)
t2.print()

t3=t2**4
t3.print()

t4=t3-t2
t4.print()

t4=t4-2
t4.print()

t4=2-t4
t4.print()