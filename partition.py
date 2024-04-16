import sys
import heapq
import numpy as np

max_iter = 1000

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
            S = S1
    return S

        
def hillClimb(A):
    S = np.random.choice([-1, 1], size=len(A))
    for x in range(max_iter):
        # Choose two distinct indices
        S1 = S.copy()
        indices_to_flip = np.random.choice(len(A), size=2, replace=False)
        # For each index, decide whether to flip the sign with a probability of 1/2
        for index in indices_to_flip:
            if np.random.rand() < 0.5:
                S1[index] *= -1
        if  residue(S1,A) < residue(S,A):
            S = S1
    return S


def simulatedAnnealing(A):
    S = np.random.choice([-1, 1], size=len(A))
    S2 = S.copy()
    for i in range(1,max_iter+1):
        # Choose two distinct indices
        S1 = S.copy()
        indices_to_flip = np.random.choice(len(A), size=2, replace=False)
        # For each index, decide whether to flip the sign with a probability of 1/2
        for index in indices_to_flip:
            if np.random.rand() < 0.5:
                S1[index] *= -1
        if  residue(S1,A) < residue(S,A):
            S = S1
        else:
            probability = np.exp((-residue(S1,A) - residue(S,A)) / T(i))
            if np.random.rand() < probability:
                S = S1
        if residue(S,A) < residue(S2,A):
            S2 = S
    return S2

def T(i):
    return (10**10) * (0.8)**(np.floor(i/300))

def residue(S, A):
    return abs(np.dot(S,A))

def PPrepeatedRand(A):
    return A
def PPhillClimb(A):
    return A
def PPsimulatedAnnealing(A):
    return A

if __name__ == "__main__":
    main()

