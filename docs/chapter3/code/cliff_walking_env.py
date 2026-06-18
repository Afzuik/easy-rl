"""
第 3 章 表格型方法 —— 悬崖行走环境（cliff walking environment）

对应书中图 3.9 的悬崖行走问题：
  
  0   1   2   3   4   5   6   7   8   9  10  11
  .   .   .   .   .   .   .   .   .   .   .   .
  .   .   .   .   .   .   .   .   .   .   .   .
  .   .   .   .   .   .   .   .   .   .   .   .
  S   C   C   C   C   C   C   C   C   C   C   G

  图例:
    S = 起点 (start)，坐标 (3, 0)
    G = 终点 (goal)，坐标 (3, 11)
    C = 悬崖 (cliff)，坐标 (3, 1) ~ (3, 10)
    . = 安全格子

动作空间: 0=上, 1=右, 2=下, 3=左
奖励:
    - 每走一步: -1
    - 掉入悬崖: -100，回到起点 S
    - 到达终点: 游戏结束
"""


class CliffWalkingEnv:
    """
    悬崖行走环境，12×4 网格。

    状态编码：state_idx = row × 12 + col（0 ~ 47）
    动作: 0=上(↑), 1=右(→), 2=下(↓), 3=左(←)
    """

    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.n_states = self.rows * self.cols
        self.n_actions = 4

        # 关键位置坐标
        self.start = (3, 0)       # 起点
        self.goal = (3, 11)       # 终点
        self.cliff_rows = set(range(3, 4))  # 悬崖所在的第4行（0-based）

        # 动作: (行偏移, 列偏移)
        self.actions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.action_symbols = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        self.action_names = {0: "up", 1: "right", 2: "down", 3: "left"}

        # 可视化网格
        self.grid_symbols = {}
        for c in range(1, 11):
            self.grid_symbols[(3, c)] = "C"
        self.grid_symbols[self.start] = "S"
        self.grid_symbols[self.goal] = "G"

    def reset(self):
        """重置环境，返回起始状态。"""
        return self._coord_to_idx(self.start)

    def step(self, state_idx, action):
        """
        执行动作，返回 (next_state, reward, done)。

        参数:
            state_idx: 当前状态索引 (0~47)
            action:    动作编号 (0~3)

        返回:
            next_state: 下一状态索引
            reward:     即时奖励
            done:       是否到达终点
        """
        if state_idx == self._coord_to_idx(self.goal):
            return state_idx, 0.0, True  # 已在终点

        r, c = self._idx_to_coord(state_idx)
        dr, dc = self.actions[action]
        nr, nc = r + dr, c + dc

        # 出界 → 留在原地
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            return state_idx, -1.0, False

        # 掉入悬崖 → 惩罚 -100，回到起点
        if self._is_cliff(nr, nc):
            return self._coord_to_idx(self.start), -100.0, False

        # 到达终点 → 游戏结束
        if (nr, nc) == self.goal:
            return self._coord_to_idx(self.goal), -1.0, True

        # 正常移动
        return self._coord_to_idx((nr, nc)), -1.0, False

    def _is_cliff(self, r, c):
        """判断 (r, c) 是否是悬崖。"""
        return r == 3 and 1 <= c <= 10

    def _coord_to_idx(self, coord):
        """(row, col) → 状态索引。"""
        r, c = coord
        return r * self.cols + c

    def _idx_to_coord(self, idx):
        """状态索引 → (row, col)。"""
        return divmod(idx, self.cols)

    def print_grid(self, values_or_policy=None, title=""):
        """
        打印网格。
        如果传入 values_or_policy 是 Q 值聚合的 V 值 → 显示数字；
        如果传入 values_or_policy 是策略数组 → 显示箭头；
        否则显示布局。
        """
        print(f"\n{title}")
        print("=" * 60)

        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                symbol = self.grid_symbols.get((r, c), ".")
                if values_or_policy is not None and (r, c) not in [self.start, self.goal] \
                        and not self._is_cliff(r, c):
                    idx = self._coord_to_idx((r, c))
                    val = values_or_policy[idx]
                    if isinstance(val, (int, float, np.floating)):
                        row_str += f"{val:>6.1f}"
                    else:
                        row_str += f"{str(val):>6s}"
                else:
                    row_str += f"{symbol:>6s}"
            print(row_str)
        print("=" * 60)

    def print_policy(self, policy, title=""):
        """打印策略：用箭头显示。"""
        print(f"\n{title}")
        print("=" * 60)
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                symbol = self.grid_symbols.get((r, c), ".")
                if (r, c) in [self.goal]:
                    row_str += f"{'G':>6s}"
                elif (r, c) in [self.start]:
                    row_str += f"{'S':>6s}"
                elif self._is_cliff(r, c):
                    row_str += f"{'C':>6s}"
                else:
                    idx = self._coord_to_idx((r, c))
                    row_str += f"{self.action_symbols[policy[idx]]:>6s}"
            print(row_str)
        print("=" * 60)


import numpy as np
