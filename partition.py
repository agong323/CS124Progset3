import sys
import heapq
import numpy as np

max_iter = 25000

def main():
    cli = sys.argv
    if len(cli) != 4:
        "bad cli"
        return
    flag = int(cli[1])
    alg = int(cli[2])
    A = []
    if flag == 0: #given input
        with open(sys.argv[3], "r") as input:
            A = [int(line.strip()) for line in input]
    if flag == 1: #testing
        A = [10, 8, 7, 6, 5]
    match alg: 
        case 0:
            print(KK(A))
        case 1:
            print(repeatedRand(A))
        case 2:
            print(hillClimb(A))
        case 3:
            print(simulatedAnnealing(A))
        case 11:
            print(PPrepeatedRand(A))
        case 12:
            print(PPhillClimb(A))
        case 13:
            print(PPsimulatedAnnealing(A))



def KK(A):
    maxheap = [-a for a in A]
    heapq.heapify(maxheap)

    while len(maxheap) > 1:
        l1 = -heapq.heappop(maxheap)
        l2 = -heapq.heappop(maxheap)       
        difference = abs(l1 - l2)
        heapq.heappush(maxheap, -difference)

    residue = -maxheap.pop()
    return residue

def repeatedRand(A):
    S = np.random.choice([-1, 1], size=len(A))
    for x in range(max_iter):
        S1 = np.random.choice([-1, 1], size=len(A))
        if  residue(S1,A) < residue(S,A):
            S = S1.copy()
    return residue(S, A)

        
def hillClimb(A):
    S = np.random.choice([-1, 1], size=len(A))
    for x in range(max_iter):
        S1 = S.copy()
        indices_to_flip = np.random.choice(len(S1), size=2, replace=False)
        S1[indices_to_flip[0]] *= -1
        if np.random.rand() < 0.5:
            S1[indices_to_flip[1]] *= -1
        if  residue(S1,A) < residue(S,A):
            S = S1.copy()
    return residue(S, A)


def simulatedAnnealing(A):
    S = np.random.choice([-1, 1], size=len(A))
    S2 = S.copy()
    for i in range(1,max_iter+1):
        S1 = S.copy()
        indices_to_flip = np.random.choice(len(S1), size=2, replace=False)
        S1[indices_to_flip[0]] *= -1
        if np.random.rand() < 0.5:
            S1[indices_to_flip[1]] *= -1
        if  residue(S1,A) < residue(S,A):
            S = S1.copy()
        else:
            probability = np.exp(-(residue(S1,A) - residue(S,A)) / T(i))
            if np.random.rand() < probability:
                S = S1.copy()
        if residue(S,A) < residue(S2,A):
            S2 = S.copy()
    return residue(S2, A)

def T(i):
    return (10**10) * (0.8)**(np.floor(i/300))

def PPrepeatedRand(A):
    n = len(A)
    P = np.array([np.random.randint(1, n+1) for _ in range(n)])
    for _ in range(max_iter):
        P1 = np.array([np.random.randint(1, n+1) for _ in range(n)])
        if  PPtransform(P1,A) < PPtransform(P,A):
            P = P1.copy()
    return PPtransform(P,A)


def PPhillClimb(A):
    n = len(A)
    P = np.array([np.random.randint(1, n+1) for _ in range(n)])
    for _ in range(max_iter):
        P1 = P.copy()
        index_to_change = np.random.randint(n)
        rand_num = np.random.randint(1, n+1)
        while rand_num == P[index_to_change]:
            rand_num = np.random.randint(1, n+1)
        P1[index_to_change] = rand_num
        if  PPtransform(P1,A) < PPtransform(P,A):
            P = P1.copy()
    return PPtransform(P,A)

def PPsimulatedAnnealing(A):
    n = len(A)
    P = np.array([np.random.randint(1, n+1) for _ in range(n)])
    P2 = P.copy()
    for i in range(1, max_iter + 1):
        P1 = P.copy()
        index_to_change = np.random.randint(n)
        rand_num = np.random.randint(1, n+1)
        while rand_num == P[index_to_change]:
            rand_num = np.random.randint(1, n+1)
        P1[index_to_change] = rand_num
        if PPtransform(P1,A) < PPtransform(P,A):
            P = P1.copy()
        else:
            probability = np.exp(-(PPtransform(P1,A) - PPtransform(P,A)) / T(i))
            if np.random.rand() < probability:
                P = P1.copy()
        if PPtransform(P,A) < PPtransform(P2,A):
            P2 = P.copy()
    return PPtransform(P2,A)

def residue(S, A):
    return abs(np.dot(S,A))

def PPtransform(P, A):
    n = len(A)
    A1 = np.zeros(n,dtype=int)
    for j in range(n):
        A1[P[j] - 1] += A[j]
    return KK(A1)

if __name__ == "__main__":
    main()



