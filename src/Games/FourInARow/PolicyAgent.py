def board_to_number(board):
    board_base3 = (board + 1).flatten()
    number = 0
    base = 3
    coeff = 1
    for v in board_base3:
        number += int(v) * coeff
        coeff *= base
    return number


class PolicyAgent:
    def __init__(self, name, cols, rows, policy, print_action=False):
        self.name = name
        self.cols = cols
        self.rows = rows
        self.policy = policy
        self.print_action = print_action

    def init(self):
        pass

    def next_turn(self, state):
        action = self.policy.get_action(board_to_number(state))
        if self.print_action:
            print(repr(action))
        return action
