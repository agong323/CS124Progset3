import sys
import heapq
import numpy as np
import matplotlib.pyplot as plt

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
    if flag == 2:
        KarmKarp = []
        RepRand = []
        HC = []
        SA = []
        PPRR = []
        PPHC = []
        PPSA = []
        for instance in range(50):
            A = np.array([np.random.randint(1, (10**12)+1) for _ in range(100)])
            KarmKarp.append(KK(A))
            RepRand.append(repeatedRand(A))
            HC.append(hillClimb(A))
            SA.append(simulatedAnnealing(A))
            PPRR.append(PPrepeatedRand(A))
            PPHC.append(PPhillClimb(A))
            PPSA.append(PPsimulatedAnnealing(A))
        plot_results(KarmKarp, "Karmarkar-Karp")
        plot_results(RepRand, "Repeated Random")
        plot_results(HC, "Hill Climb")
        plot_results(SA, "Simulated Annealing")
        plot_results(PPRR, "Prepartitioned Repeated Random")
        plot_results(PPHC, "Prepartitioned Hill Climb")
        plot_results(PPSA, "Prepartitioned Simulated Annealing")

    else: 
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


def plot_results(data, title):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    ax[0].bar(range(len(data)), data, color='skyblue')
    ax[0].set_title(f'Bar Chart of {title}')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Value')
    
    # Box plot
    ax[1].boxplot(data)
    ax[1].set_title(f'Box Plot of {title}')
    ax[1].set_xlabel('Data')
    ax[1].set_ylabel('Value')
    
    plt.show()

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
    return residue(S, A)

        
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
    return residue(S, A)


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
    return residue(S2, A)

def T(i):
    return (10**10) * (0.8)**(np.floor(i/300))

def PPrepeatedRand(A):
    n = len(A)
    P = np.array([np.random.randint(1, n+1) for _ in range(n)])
    Aprime = PPtransform(P,A)
    for _ in range(max_iter):
        P1 = np.array([np.random.randint(1, n+1) for _ in range(n)])
        A1prime = PPtransform(P1,A)
        if  KK(A1prime) < KK(Aprime):
            Aprime = A1prime
    return KK(Aprime)


def PPhillClimb(A):
    n = len(A)
    P = np.array([np.random.randint(1, n+1) for _ in range(n)])
    Aprime = PPtransform(P,A)
    for _ in range(max_iter):
        P1 = P.copy()
        index_to_change = np.random.randint(n)
        rand_num = np.random.randint(1, n+1)
        while rand_num == P[index_to_change]:
            rand_num = np.random.randint(1, n+1)
        P1[index_to_change] = rand_num
        A1prime = PPtransform(P1,A)
        if  KK(A1prime) < KK(Aprime):
            Aprime = A1prime
    return KK(Aprime)

def PPsimulatedAnnealing(A):
    n = len(A)
    P = np.array([np.random.randint(1, n+1) for _ in range(n)])
    Aprime = PPtransform(P,A)
    A2prime = Aprime.copy()
    for i in range(1, max_iter + 1):
        P1 = P.copy()
        index_to_change = np.random.randint(n)
        rand_num = np.random.randint(1, n+1)
        while rand_num == P[index_to_change]:
            rand_num = np.random.randint(1, n+1)
        P1[index_to_change] = rand_num
        A1prime = PPtransform(P1,A)
        if  KK(A1prime) < KK(Aprime):
            Aprime = A1prime
        elif np.random.rand() < np.exp((-KK(A1prime) - KK(Aprime)) / T(i)):
            Aprime = A1prime
        if KK(Aprime) < KK(A2prime):
            A2prime = Aprime
    return KK(A2prime)

def residue(S, A):
    return abs(np.dot(S,A))

def PPtransform(P, A):
    n = len(A)
    A1 = np.zeros(n,dtype=int)
    for j in range(n):
        A1[P[j] - 1] += A[j]
    return A1

if __name__ == "__main__":
    main()

