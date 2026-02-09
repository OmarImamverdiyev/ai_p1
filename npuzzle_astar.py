"""
Ahmadov Kamal
Imamverdiyev Omar

CSCI 6511 Artificial Intelligence
Project 1 — Sub Project C: N-Puzzle (A* Search)

Overview:
  This program solves the N-Puzzle on an n x n grid (3 <= n <= 8) using the A* search algorithm.
  Tiles are numbered 1..(n^2-1) with one blank space represented by 0. A move slides one tile
  horizontally or vertically into the blank.

Input Format:
  - A text file containing n lines with n entries per line.
  - Entries are typically tab-separated (as provided by the instructor).
  - The blank space may appear as:
      (1) an explicit 0, OR
      (2) an empty cell in a tab-delimited row, OR
      (3) a missing column in a space-aligned row (handled by our parser).

Output:
  - Minimum number of moves to reach the goal state
  - A move sequence using:
      U = move blank up
      D = move blank down
      L = move blank left
      R = move blank right
  - Optional: print all intermediate boards with --show

How to Run:
  python3 npuzzle_astar.py <input_file>
  python3 npuzzle_astar.py <input_file> --show

Goal State:
  1  2  3  ...  n
  ...
  (n^2-1)  0
"""


from __future__ import annotations
import argparse
import heapq
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import time


Move = str  # "U", "D", "L", "R"
Board = Tuple[int, ...]


def read_board(path: str) -> Tuple[int, Board]:
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]

    n = len(raw_lines)
    if not (3 <= n <= 8):
        raise ValueError(f"n must be between 3 and 8, got n={n}")

    # ---------- Case 1: TAB-delimited (preserve empty cells) ----------
    # If there are tabs, parse by '\t' so empty tokens survive.
    if any("\t" in ln for ln in raw_lines):
        rows: List[List[int]] = []
        for ln in raw_lines:
            parts = ln.split("\t")
            # If teacher used tabs, we expect exactly n columns.
            # If not, we'll fall back to space-aligned parsing below.
            if len(parts) != n:
                rows = []
                break
            row: List[int] = []
            for cell in parts:
                cell = cell.strip()
                row.append(0 if cell == "" else int(cell))
            rows.append(row)

        if rows:
            flat = [x for r in rows for x in r]
            expected = set(range(n * n))
            if set(flat) != expected:
                raise ValueError(f"Board must contain all numbers 0..{n*n-1} exactly once.")
            return n, tuple(flat)

    # ---------- Case 2: SPACE-ALIGNED / FIXED-COLUMN parsing ----------
    # Extract numbers with their character start positions.
    # If one row is missing the blank, it will have n-1 numbers.
    num_pat = re.compile(r"\d+")
    row_matches = []
    counts = []

    for ln in raw_lines:
        matches = [(m.group(0), m.start()) for m in num_pat.finditer(ln)]
        row_matches.append(matches)
        counts.append(len(matches))

    max_count = max(counts)
    # If all rows already have n numbers, just parse them normally.
    if max_count == n and all(c == n for c in counts):
        rows = [[int(tok) for tok, _ in matches] for matches in row_matches]
        flat = [x for r in rows for x in r]
        expected = set(range(n * n))
        if set(flat) != expected:
            raise ValueError(f"Board must contain all numbers 0..{n*n-1} exactly once.")
        return n, tuple(flat)

    # Otherwise, we expect exactly one row to have n-1 numbers (the blank row),
    # and the rest to have n numbers.
    if max_count != n or not all(c in (n, n - 1) for c in counts) or counts.count(n - 1) != 1:
        # Last fallback: plain whitespace split (requires explicit 0)
        rows_ws: List[List[int]] = []
        for ln in raw_lines:
            parts = ln.split()
            rows_ws.append([int(x) for x in parts])
        if any(len(r) != n for r in rows_ws):
            raise ValueError(
                "Could not parse as tab-delimited or space-aligned grid. "
                "If using spaces, the file must be column-aligned; otherwise include 0 for blank."
            )
        flat = [x for r in rows_ws for x in r]
        expected = set(range(n * n))
        if set(flat) != expected:
            raise ValueError(f"Board must contain all numbers 0..{n*n-1} exactly once.")
        return n, tuple(flat)

    # Choose an "anchor" row that has n numbers to infer column start positions.
    anchor_idx = counts.index(n)
    anchors = [pos for _, pos in row_matches[anchor_idx]]

    # Tolerance for aligning numbers to anchors
    diffs = [anchors[i + 1] - anchors[i] for i in range(n - 1)]
    min_step = min(diffs) if diffs else 2
    tol = max(1, min_step // 2)

    def fill_row(matches: List[Tuple[str, int]]) -> List[int]:
        # If already complete, take tokens in order
        if len(matches) == n:
            return [int(tok) for tok, _ in matches]

        # If missing one cell, insert 0 at the missing anchor position.
        out: List[int] = []
        j = 0  # index into matches
        for i in range(n):  # for each anchor/column
            if j >= len(matches):
                out.append(0)
                continue
            tok, pos = matches[j]
            if abs(pos - anchors[i]) <= tol:
                out.append(int(tok))
                j += 1
            else:
                out.append(0)
        if len(out) != n:
            raise ValueError("Internal parse error while reconstructing missing blank.")
        return out

    rows = [fill_row(m) for m in row_matches]
    flat = [x for r in rows for x in r]

    expected = set(range(n * n))
    if set(flat) != expected:
        # Helpful diagnostic
        missing = sorted(expected - set(flat))
        extra = sorted(set(flat) - expected)
        raise ValueError(
            f"Parsed grid, but numbers are wrong. Missing={missing}, Extra={extra}. "
            "Your file may not be consistently column-aligned."
        )

    return n, tuple(flat)

def goal_board(n: int) -> Board:
    # 1..N then 0
    return tuple(list(range(1, n * n)) + [0])


def inversion_count(seq: List[int]) -> int:
    # Count inversions ignoring 0
    arr = [x for x in seq if x != 0]
    inv = 0
    for i in range(len(arr)):
        ai = arr[i]
        for j in range(i + 1, len(arr)):
            if ai > arr[j]:
                inv += 1
    return inv


def is_solvable(n: int, b: Board) -> bool:
    inv = inversion_count(list(b))
    if n % 2 == 1:
        # odd grid: solvable if inversions even
        return inv % 2 == 0

    # even grid:
    # row of blank counted from bottom starting at 1
    blank_idx = b.index(0)
    blank_row_from_top = blank_idx // n  # 0-based
    blank_row_from_bottom = n - blank_row_from_top  # 1..n
    # solvable iff:
    #   blank on even row from bottom and inversions odd
    #   OR blank on odd row from bottom and inversions even
    if blank_row_from_bottom % 2 == 0:
        return inv % 2 == 1
    else:
        return inv % 2 == 0


def build_goal_positions(n: int) -> List[Tuple[int, int]]:
    """
    goal_pos[val] = (row, col) for val in 0..n*n-1
    """
    pos = [(0, 0)] * (n * n)
    # 1..N
    for val in range(1, n * n):
        idx = val - 1
        pos[val] = (idx // n, idx % n)
    # 0 at the end
    pos[0] = (n - 1, n - 1)
    return pos


def manhattan(n: int, b: Board, goal_pos: List[Tuple[int, int]]) -> int:
    dist = 0
    for idx, val in enumerate(b):
        if val == 0:
            continue
        r, c = divmod(idx, n)
        gr, gc = goal_pos[val]
        dist += abs(r - gr) + abs(c - gc)
    return dist


def linear_conflict(n: int, b: Board, goal_pos: List[Tuple[int, int]]) -> int:
    """
    Linear conflict heuristic component:
    Adds 2 moves for each pair of tiles in the same row/col whose goal positions
    are in that row/col but reversed relative order.

    Total heuristic = Manhattan + linear_conflict
    (still admissible)
    """
    conflict = 0

    # Row conflicts
    for r in range(n):
        row_tiles: List[int] = []
        for c in range(n):
            val = b[r * n + c]
            if val != 0 and goal_pos[val][0] == r:
                row_tiles.append(val)
        # count inversions by goal column among these tiles
        for i in range(len(row_tiles)):
            gi = goal_pos[row_tiles[i]][1]
            for j in range(i + 1, len(row_tiles)):
                gj = goal_pos[row_tiles[j]][1]
                if gi > gj:
                    conflict += 2

    # Column conflicts
    for c in range(n):
        col_tiles: List[int] = []
        for r in range(n):
            val = b[r * n + c]
            if val != 0 and goal_pos[val][1] == c:
                col_tiles.append(val)
        # count inversions by goal row among these tiles
        for i in range(len(col_tiles)):
            gi = goal_pos[col_tiles[i]][0]
            for j in range(i + 1, len(col_tiles)):
                gj = goal_pos[col_tiles[j]][0]
                if gi > gj:
                    conflict += 2

    return conflict


def heuristic(n: int, b: Board, goal_pos: List[Tuple[int, int]]) -> int:
    return manhattan(n, b, goal_pos) + linear_conflict(n, b, goal_pos)


def neighbors(n: int, b: Board) -> Iterable[Tuple[Board, Move]]:
    z = b.index(0)
    zr, zc = divmod(z, n)

    def swap_and_yield(nidx: int, mv: Move):
        bb = list(b)
        bb[z], bb[nidx] = bb[nidx], bb[z]
        return (tuple(bb), mv)

    # Moves: U means blank moves up (tile moves down into blank),
    # but move naming is conventional; we’ll use direction of blank movement.
    if zr > 0:
        yield swap_and_yield(z - n, "U")
    if zr < n - 1:
        yield swap_and_yield(z + n, "D")
    if zc > 0:
        yield swap_and_yield(z - 1, "L")
    if zc < n - 1:
        yield swap_and_yield(z + 1, "R")


@dataclass(order=True)
class PQItem:
    f: int
    h: int
    g: int
    board: Board


def astar(n: int, start: Board, use_heuristic: bool = True):

    goal = goal_board(n)
    if start == goal:
        return 0, [], [start]

    goal_pos = build_goal_positions(n)

    # For path reconstruction
    parent: Dict[Board, Tuple[Optional[Board], Optional[Move]]] = {start: (None, None)}
    g_best: Dict[Board, int] = {start: 0}

    expanded = 0
    max_frontier = 0
    start_time = time.time()

    h0 = heuristic(n, start, goal_pos) if use_heuristic else 0

    pq: List[PQItem] = []
    heapq.heappush(pq, PQItem(f=h0, h=h0, g=0, board=start))

    while pq:
        cur = heapq.heappop(pq)
        b = cur.board
        
        # Skip stale PQ entries
        if cur.g != g_best.get(b, 10**18):
            continue

        expanded += 1
        max_frontier = max(max_frontier, len(pq))

        if b == goal:
            elapsed = time.time() - start_time
            # reconstruct
            moves: List[Move] = []
            boards: List[Board] = []
            node: Optional[Board] = b
            while node is not None:
                boards.append(node)
                p, mv = parent[node]
                if mv is not None:
                    moves.append(mv)
                node = p
            boards.reverse()
            moves.reverse()

            return {
                "moves": cur.g,
                "path": moves,
                "boards": boards,
                "expanded": expanded,
                "max_frontier": max_frontier,
                "time": elapsed,
            }

        for nb, mv in neighbors(n, b):
            ng = cur.g + 1
            old = g_best.get(nb)
            if old is None or ng < old:
                g_best[nb] = ng
                parent[nb] = (b, mv)
                nh = heuristic(n, nb, goal_pos) if use_heuristic else 0
                heapq.heappush(pq, PQItem(f=ng + nh, h=nh, g=ng, board=nb))

    raise RuntimeError("No solution found (this should not happen if solvable).")
    


def format_board(n: int, b: Board) -> str:
    w = len(str(n * n - 1))
    out_lines = []
    for r in range(n):
        row = b[r * n : (r + 1) * n]
        out_lines.append(" ".join((" " * w if x == 0 else str(x).rjust(w)) for x in row))
    return "\n".join(out_lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Solve n-puzzle using A* (Manhattan + Linear Conflict).")
    ap.add_argument("file", help="Path to input file")
    ap.add_argument("--show", action="store_true", help="Print boards along the solution path.")
    ap.add_argument("--evaluation", action="store_true",
                help="Compare UCS (h=0) with A* heuristic")
    args = ap.parse_args()

    n, start = read_board(args.file)

    print(f"n = {n}")
    print("Start:")
    print(format_board(n, start))
    print()

    if not is_solvable(n, start):
        print("This puzzle configuration is NOT solvable.")
        return

    if args.evaluation:
        print("Running Uniform Cost Search (h = 0)...")
        ucs = astar(n, start, use_heuristic=False)

        print("Running A* with heuristic...")
        astar_h = astar(n, start, use_heuristic=True)

        print("\n=== Evaluation Results ===")
        print("UCS (no heuristic):")
        print(f"  Expanded states: {ucs['expanded']}")
        print(f"  Max frontier size: {ucs['max_frontier']}")
        print(f"  Runtime: {ucs['time']:.3f} seconds")

        print("\nA* with heuristic:")
        print(f"  Expanded states: {astar_h['expanded']}")
        print(f"  Max frontier size: {astar_h['max_frontier']}")
        print(f"  Runtime: {astar_h['time']:.3f} seconds")

        print("\n=== Solution (A* with heuristic) ===")
        print(f"Minimum moves: {astar_h['moves']}")
        print("Move sequence:", "".join(astar_h["path"]))

        if args.show:
            for i, b in enumerate(astar_h["boards"]):
                print(f"\nStep {i}:")
                print(format_board(n, b))

    else:
        # NORMAL MODE: A* only
        result = astar(n, start, use_heuristic=True)

        print(f"Minimum moves: {result['moves']}")
        print("Move sequence:", "".join(result["path"]))

        if args.show:
            for i, b in enumerate(result["boards"]):
                print(f"\nStep {i}:")
                print(format_board(n, b))


if __name__ == "__main__":
    main()