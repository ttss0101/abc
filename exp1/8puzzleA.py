import heapq

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Misplaced Tiles Heuristic
def heuristic(state):
    count = 0

    for i in range(9):
        if state[i] != 0 and state[i] != GOAL[i]:
            count += 1

    return count


# Generate possible movements
def get_neighbors(state):
    neighbors = []

    blank = state.index(0)
    row = blank // 3
    col = blank % 3

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in moves:
        new_row = row + dr
        new_col = col + dc

        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_pos = new_row * 3 + new_col

            new_state = list(state)
            new_state[blank], new_state[new_pos] = (
                new_state[new_pos], new_state[blank]
            )

            neighbors.append(tuple(new_state))

    return neighbors


# A* Search Algorithm
def astar(start):
    pq = []
    heapq.heappush(pq, (heuristic(start), 0, start, [start]))

    best_g = {start: 0}
    counter = 0

    while pq:
        f, g, current, path = heapq.heappop(pq)

        if current == GOAL:
            return path

        if g > best_g[current]:
            continue

        for next_state in get_neighbors(current):
            new_g = g + 1

            if next_state not in best_g or new_g < best_g[next_state]:
                best_g[next_state] = new_g

                new_h = heuristic(next_state)
                new_f = new_g + new_h

                counter += 1

                heapq.heappush(
                    pq,
                    (new_f, new_g, next_state, path + [next_state])
                )

    return None


# Main Program
# Main Program

print("Enter the 8 Puzzle (use 0 for blank space)")
print("Enter 9 numbers separated by spaces:")

start = tuple(map(int, input().split()))

if len(start) != 9 or sorted(start) != list(range(9)):
    print("Invalid input! Enter numbers 0 to 8 exactly once.")

else:

    print("\nInitial State:")

    for i in range(0, 9, 3):
        print(start[i:i+3])

    print("\nHeuristic Value:", heuristic(start))

    solution = astar(start)

    if solution:

        print("\nSolution Path:")

        for step, state in enumerate(solution):

            print("\nStep:", step)

            for i in range(0, 9, 3):
                print(state[i:i+3])

        print("\nTotal Moves:", len(solution) - 1)

    else:
        print("No solution found")