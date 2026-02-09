# CSCI 6511 – Artificial Intelligence

## Project 1, Sub Project C: N-Puzzle Solver (A* Search)

### Authors

* **Kamal Ahmadov** - kamal.ahmadov@gwu.edu
* **Omar Imamverdiyev** - o.imamverdiyev@gwu.edu

### Instructor

* **Dr. Amrinder Arora**

---

## 1. Problem Description

The **N-Puzzle** is a sliding tile puzzle consisting of an `n × n` grid containing tiles numbered `1` through `n²−1` and one blank space. A move consists of sliding a tile **horizontally or vertically** into the blank space.

Common examples:

* **8-Puzzle** → `3 × 3` grid
* **15-Puzzle** → `4 × 4` grid
* **24-Puzzle** → `5 × 5` grid

The goal is to rearrange the tiles into increasing numerical order with the blank tile in the bottom-right corner, using the **fewest possible moves**.

This project implements an **optimal solver** using the **A*** search algorithm.

---

## 2. Constraints

* `3 ≤ n ≤ 8`
* Exactly one blank space
* Tiles may move **up, down, left, or right**
* Input file may contain:

  * Explicit `0` for the blank, **or**
  * An empty cell in a tab-delimited row, **or**
  * A missing column in a space-aligned row (as provided by the instructor)

---

## 3. Approach & Algorithm

### 3.1 A* Search

We use **A*** search to guarantee an **optimal (minimum-move) solution**.

For each board state:

* **g(n)** = number of moves taken so far
* **h(n)** = heuristic estimate to the goal
* **f(n) = g(n) + h(n)**

The state with the smallest `f(n)` is expanded first.

---

### 3.2 Heuristic Function

We use an **admissible and consistent heuristic**:

#### Manhattan Distance

Sum of the vertical and horizontal distances of each tile from its goal position.

#### Linear Conflict (Improvement)

Adds an extra cost when two tiles are in the same row or column but reversed relative to their goal order.

```
h(n) = Manhattan Distance + Linear Conflict
```

This heuristic:

* Never overestimates the true cost
* Significantly improves performance over Manhattan alone

---

### 3.3 Solvability Check

Before running A*, the program checks if the puzzle is solvable:

* For **odd grid sizes**: number of inversions must be even
* For **even grid sizes**: solvability depends on inversion count and the blank row position (from bottom)

Unsolvable puzzles are detected immediately and reported.

---

## 4. Input Format

A text file containing `n` rows and `n` columns.

### Valid Input Examples

#### Tab-Delimited (with blank)

```
22	23		24	20
```

#### Space-Aligned (blank as missing column)

```
22  23      24  20
```

#### Explicit Zero

```
22 23 0 24 20
```

The parser automatically detects and handles all supported formats.

---

## 5. Output

The program prints:

1. Puzzle size (`n`)
2. Initial board configuration
3. Minimum number of moves
4. Move sequence using:

   * `U` → blank moves up
   * `D` → blank moves down
   * `L` → blank moves left
   * `R` → blank moves right
5. (Optional) Full solution path when `--show` is used

### Example Output

```
n = 5
Start:
11  1  2  3 14
12  7  9 10 13
 6  8 18  5  4
21 16 17 19 15
22 23    24 20

Minimum moves: 38
Move sequence: LLURRULLUURRRDDRUULDDRUULDLLDLURRDRRDD
```

---

## 6. How to Run

### Requirements

* Python **3.10+** (tested with Python 3.12)
* No external libraries required

### Run Command

```bash
python3 npuzzle_astar.py <input_file>
```

### Show Full Solution Path

```bash
python3 npuzzle_astar.py <input_file> --show
```

### Run in Evaluation Mode

To compare Uniform Cost Search (UCS) and A* search with heuristic guidance, the program can be executed in evaluation mode:

```bash
python3 npuzzle_astar.py <input_file> --evaluation
```
>Note: In report, we tested on `eval_test.txt`
---

## 7. Design Decisions

* **Tuple-based board representation** for fast hashing
* **Priority queue (heap)** for A* frontier
* **Parent map** for efficient path reconstruction
* Robust input parser to match instructor-provided files
* Early solvability detection to avoid wasted computation

---

## 8. Performance Notes

* A* performs well for **3×3** and **4×4** puzzles
* **5×5 and larger puzzles** may take significant time and memory
* This is expected behavior due to the exponential state space
* The heuristic used is optimal under assignment constraints

---

## 9. Limitations & Future Work

* Large puzzles (`n ≥ 6`) may exceed memory/time limits
* Future improvements could include:

  * IDA* (Iterative Deepening A*)
  * Pattern database heuristics
  * Bidirectional search

---

## 10. Conclusion

This project demonstrates:

* Correct implementation of A* search
* Use of admissible heuristics
* Robust input handling
* Guaranteed optimal solutions for solvable N-Puzzle instances

