#!/usr/bin/env python
# coding: utf-8

# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer, IBMQ
from qiskit.quantum_info.operators import Operator
import numpy as np
import math


# In[32]:


def mcx(n):  #total n qubits
    matrix = np.identity(2**n, dtype = int)
    matrix[2**(n-1) - 1][2**(n-1) - 1] = 0
    matrix[2**(n-1) - 1][2**n - 1] = 1
    matrix[2**n - 1][2**(n-1) - 1] = 1
    matrix[2**n - 1][2**n - 1] = 0
    return Operator(matrix)


# In[33]:


def mark(bits, circuit, target):
    n = len(bits)
    texts = bits[::-1]
    ctrls = []
    for i in range(n):
        if texts[i] == "0":
            circuit.x(i)
            ctrls.append(i)
        elif texts[i] == "1":
            ctrls.append(i)
    mymcx = mcx(len(ctrls)+1)
    circuit.append(mymcx, ctrls+[target])
    for i in range(n):
        if texts[i] == "0":
            circuit.x(i)
    qc.barrier()


# In[34]:


def code(texts, n):
    output = []
    for t in texts:
        string = t.replace("C", "00")
        string = string.replace("=O", "01")
        string = string.replace("O=", "01")
        string = string.replace("O", "10")
        output.append(string)

    output = [bits + "1"*(n-len(bits)) for bits in output]
    return output


# In[35]:


def expand(texts):
    if len(texts) == 1:
        return texts
    if len(texts) == 2:
        return [text1 + text2 for text1 in texts[0] for text2 in texts[1]]
    if len(texts) > 2:
        return [text1 + text2 for text1 in texts[0] for text2 in expand(texts[1:])]


# In[36]:


p1 = [["CC"],
      ["C", "CO"]
    ]
p2 = [["O"],
      ["CC", "CCC", "=C"]
     ]

b1, b2 = [], [] 
t1 = expand(p1) + [text[::-1] for text in expand(p1)]
t2 = expand(p2) + [text[::-1] for text in expand(p2)]
for x in t1:
    if x not in b1:
        b1.append(x)
for x in t2:
    if x not in b2:
        b2.append(x)


# In[37]:


p1 = [["CC"],
      ["C", "CO"]
    ]
p2 = [["O"],
      ["CC", "CCC", "=C"]
     ]

b1, b2 = [], [] 
t1 = expand(p1) + [text[::-1] for text in expand(p1)]
t2 = expand(p2) + [text[::-1] for text in expand(p2)]
for x in t1:
    if x not in b1:
        b1.append(x)
for x in t2:
    if x not in b2:
        b2.append(x)

n_data = max(len(text) for text in b1 + b2)*2
bits1 = code(b1, n_data)
bits2 = code(b2, n_data)

n = n_data + 2  #n must >= 4
m = 2  #ans
loop_times = round(math.acos(math.sqrt(m/(2**(n-2))))/(2 * math.sqrt(m/(2**(n-2)))))


# In[38]:


q = QuantumRegister (n)
c = ClassicalRegister (n)
qc = QuantumCircuit (q, c)

#initiation
for i in range(n-2):
    qc.h(i)

#Grover's iteration
for i in range(loop_times):
    #Oracles
    for bits in bits1:       #Mark patent 1
        mark(bits, qc, n-2)
    for bits in bits2:       #Mark patent 2
        mark(bits, qc, n-1)
    qc.cz(n-2, n-1)          #Flip the phase
    for bits in bits2[::-1]:       #Disentangle patent 2
        mark(bits, qc, n-1)
    for bits in bits1[::-1]:       #Disentangle patent 1
        mark(bits, qc, n-2)
    
    #Amplitude amplification
    for i in range(n-2):
        qc.h(i)
        qc.x(i)
    qc.h(n-3)
    mymcx = mcx(n-2)
    qc.append(mymcx, [*range(n-2)])
    qc.h(n-3)
    for i in range(n-2):
        qc.x(i)
        qc.h(i)

qc.draw()


# In[39]:


backend = Aer.get_backend('statevector_simulator')
job = execute(qc, backend)
result = job.result()
outputstate = result.get_statevector(qc, decimals=3)

# ans: CCCO, OCCC
# = 00000010, 10000000
# = 2, 128


# In[40]:


for i in range(2**10):
    print(i, outputstate[i])


# In[ ]:




