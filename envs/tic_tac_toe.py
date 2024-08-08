import random

import numpy as np

from .env import Environment, Observation, Player

# Constants for Tic-Tac-Toe game.
PLAYERS = (Player.X, Player.O)
STARTING_PLAYER = Player.X
PLAYER_ENCODING = {Player.X: "X", Player.O: "O"}
BOARD_SHAPE = (3, 3)
NUM_PLAYERS = 2
WIN_CONDITIONS = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


# Environment for Tic-Tac-Toe game.
class TicTacToe(Environment):
    def __init__(
        self,
        board_shape: tuple = BOARD_SHAPE,
        num_players: int = NUM_PLAYERS,
        players: tuple = PLAYERS,
        starting_player: Player = STARTING_PLAYER,
        enc_players: dict = PLAYER_ENCODING,
    ):
        super().__init__(board_shape, num_players, players, starting_player)
        self.enc_players = enc_players

    def reset(self):
        """Reset the environment to the initial state.

        Returns:
            Observation: Resulting observation.
        """
        self.board = np.zeros(self.board.shape, dtype=np.int8)
        self.current_player = self.starting_player
        self.valid_actions = self._update_valid_actions()
        self.winner = None
        return self.last()

    def step(self, action: int, player: Player):
        """Progressing the environment according to given action.

        Args:
            action (int): Action ID.
            player (Player): Acting player.

        Returns:
            Observation: Resulting observation.
        """
        if (self._check_done() == -2) and action != -1:
            assert player == self.current_player
            self._act(action, player)
            next_player = Player.X if self.current_player == Player.O else Player.O
            self.current_player = next_player
            return self.last()
        else:
            return

    def _check_done(self):
        """Check if the game is ended.

        Returns:
            int: Encoded game state.
        """
        flat_board = self.board.flatten()
        for cond in WIN_CONDITIONS:
            if np.all(flat_board[[cond]] == 1):
                return 1
            if np.all(flat_board[[cond]] == -1):
                return -1
        if not np.any(flat_board == 0):
            return 0
        return -2

    def _update_valid_actions(self):
        """Update valid actions (Empty tiles).

        Returns:
            list: Index of empty tiles.
        """
        return np.argwhere(self.board.flatten() == 0).flatten()

    def _act(self, pos: int, player: Player):
        """Put the given player's mark to given position.

        Args:
            pos (int): Tile ID.
            player (Player): Player class.
        """
        pos_tuple = divmod(pos, self.board_shape[0])
        if self._is_valid_action(pos_tuple):
            self.board[pos_tuple] = player.value

    def _is_valid_action(self, pos: tuple):
        """Check if given action is valid.

        Args:
            pos (tuple): Action encoded as table index.

        Returns:
            bool: Action validity.
        """
        return (self.board[pos] == 0) and (self._check_done() == -2)

    def _sample_action(self):
        """Return a random valid action.

        Returns:
            int: Action ID.
        """
        if len(self.valid_actions) > 0:
            return random.choice(self.valid_actions)
        return -1

    def render(self):
        """Format the game board as a string.

        Returns:
            str: Board formatted as string.
        """
        rendered_board = list()
        for r in range(self.board.shape[0]):
            row = self.board[r]
            row_str = " | ".join(
                [self.enc_players.get(Player(e), " ") if e != 0 else " " for e in row]
            )
            if r == 0:
                rendered_board.append("".join(["-" for _ in range(len(row_str))]))
            rendered_board.append(row_str)
            rendered_board.append("".join(["-" for _ in range(len(row_str))]))
        return "\n".join(rendered_board)

    def last(self):
        """Prepare an Observation from the current state of the environment.

        Returns:
            Observation: Resulting observation.
        """
        state = tuple(self.board.flatten())
        self.valid_actions = self._update_valid_actions()
        done = self._check_done()
        if done == -2:
            reward = {Player.X: 0, Player.O: 0}
        elif done == 0:
            reward = {Player.X: 0, Player.O: 0}
            self.winner = Player(0)
        elif done == 1:
            reward = {Player.X: 1, Player.O: -1}
            self.winner = Player(1)
        elif done == -1:
            reward = {Player.X: -1, Player.O: 1}
            self.winner = Player(-1)
        terminate = True if done != -2 else False
        return Observation(state, self.valid_actions, reward, terminate)
