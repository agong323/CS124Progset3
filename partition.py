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
    randSol = np.random.choice([-1, 1], size=len(A))
    min_residue = abs(np.dot(randSol,A))
    for x in range(max_iter):
        r = np.random.choice([-1, 1], size=len(A))
        r_res = abs(np.dot(r,A))
        if  r_res < min_residue:
            min_residue = r_res
            randSol = r
    return randSol

        
def hillClimb(A):
    randSol = np.random.choice([-1, 1], size=len(A))

    return A
def simulatedAnnealing(A):
    return A
def PPrepeatedRand(A):
    return A
def PPhillClimb(A):
    return A
def PPsimulatedAnnealing(A):
    return A

if __name__ == "__main__":
    main()

